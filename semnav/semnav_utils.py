import numpy as np
import matplotlib.pyplot as plt
import cv2
import quaternion

class DoorDetection:
    def __init__(self, camera_hfov=79, camera_width=640, camera_height=480, 
                 door_color_code=8, depth_min=0.5, depth_max=5.0, depth_threshold=4.0):
        self.depth_camera_hfov = camera_hfov
        self.depth_camera_width = camera_width
        self.depth_camera_height = camera_height
        self.depth_min = depth_min
        self.depth_max = depth_max

        self.door_color_code = door_color_code
        self.door_depth_threshold = depth_threshold
        self.depth_camera_intrinsics = self.build_intrinsic_matrix()

    def build_intrinsic_matrix(self):
        """Return depth-camera intrinsics for HM3D."""

        hfov = self.depth_camera_hfov * np.pi / 180.0
        fx = 1.0 / np.tan(hfov / 2.0)
        fy = fx * (640.0 / 480.0)
        return {
            "fx": fx,
            "fy": fy,
            "cx": 1.0,
            "cy": 1.0
        }

    def decode_depth_image(self, depth_normalized):
        """
        Decodes a normalized [0, 1] depth image into actual depth values.
        
        Args:
            depth_normalized (np.array): 480x640x1 array with values 0 to 1.
            min_depth (float): The distance represented by 0.0.
            max_depth (float): The distance represented by 1.0.
            
        Returns:
            np.array: Depth map in real-world units (e.g., meters).
        """
        # Ensure it's a float array for calculation
        depth_float = depth_normalized.astype(np.float32)
        
        # Linear mapping: Depth = min + (normalized_val * (max - min))
        actual_depth = self.depth_min + (depth_float * (self.depth_max - self.depth_min))
        
        return actual_depth

    def get_object_mask(self, img_array):
        """
        Loads an image and returns a binary mask where the pixels matching 
        the hex_color are 1 (True) and others are 0 (False).
        """
        mask = np.all(img_array == self.door_color_code, axis=-1)

        return mask.astype(np.uint8) * 255 

    def get_camera_to_world_coordinate(self, r, t):
        """
        Docstring for get_camera_to_world_coordinate
        
        :param r: rotation vector
        :param t: translation vector
        """
        rotation = quaternion.as_rotation_matrix(r)
        position = t
        camera_world = np.eye(4)
        camera_world[0:3, 0:3] = rotation
        camera_world[0:3, 3] = position
        return camera_world
     
    def project_pixel_to_global_frame(self, u, v, depth, camera_to_world):
        """
        u, v: Pixel coordinates of the door pillar
        depth: Depth value at (u, v) from depth image
        intrinsics: Dictionary with 'fx', 'fy', 'cx', 'cy'
        camera_to_world_transformation matrix (4x4)
        """
        # code borrowed from https://aihabitat.org/docs/habitat-lab/view-transform-warp.html 
        # 1. Camera Intrinsics
        fx, fy = self.depth_camera_intrinsics['fx'], self.depth_camera_intrinsics['fy']
        cx, cy = self.depth_camera_intrinsics['cx'], self.depth_camera_intrinsics['cy']

        # 2. Project to 3D Camera Frame (Standard Pinhole Model)
        # Z is depth, X and Y are calculated based on similarity
        z = depth
        x_ndc = (u / (self.depth_camera_width - 1)) * 2.0 - 1.0
        y_ndc = 1.0 - (v / (self.depth_camera_height - 1)) * 2.0

        x_cam = float(x_ndc * z / fx)
        y_cam = float(y_ndc * z / fy)
        z_cam = float(-z)  # right-handed camera, −Z forward

        # Rotate camera coordinates (x_cam, z_cam) to align with world frame
        world_h = camera_to_world @ np.array([x_cam, y_cam, z_cam, 1.0], dtype=np.float32)
        return world_h[:3] / world_h[3]

    def filter_doors_with_depth(self, door_mask, depth_map):
        """
        Advanced filter that handles 'cut-off' door frames by grouping components
        with similar depths and horizontal proximity.
        """
        # 1. Clean up noise
        kernel = np.ones((4,4), np.uint8)
        door_mask = cv2.morphologyEx(door_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)

        # 2. Get connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(door_mask, connectivity=8)
        if num_labels <=2:
            return np.zeros_like(door_mask)
        
        blobs = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < 8000: continue # Ignore tiny noise
            
            # Calculate median depth for this specific blob
            blob_depths = depth_map[labels == i]
            median_depth = np.median(blob_depths)
            
            if self.door_depth_threshold and median_depth > self.door_depth_threshold:
                continue
                
            blobs.append({
                'id': i,
                'stats': stats[i],
                'depth': median_depth,
                'centroid': centroids[i]
            })

        # 3. GROUPING LOGIC
        # We group blobs that have a similar depth (within 20cm)
        groups = []
        already_grouped = set()

        for i, b1 in enumerate(blobs):
            if b1['id'] in already_grouped: continue
            
            current_group = [b1]
            already_grouped.add(b1['id'])
            
            for j, b2 in enumerate(blobs):
                if i == j or b2['id'] in already_grouped: continue
                
                # If depths are very similar, they are likely parts of the same door
                depth_diff = abs(b1['depth'] - b2['depth'])
                if depth_diff < 0.25: # 25cm tolerance
                    current_group.append(b2)
                    already_grouped.add(b2['id'])
            
            groups.append(current_group)

        # 4. FILTER GROUPS
        door_data = []
        for group in groups:
            # Combined stats for the group
            total_area = sum(b['stats'][cv2.CC_STAT_AREA] for b in group)
            
            # Combined Bounding Box
            min_x = min(b['stats'][cv2.CC_STAT_LEFT] for b in group)
            min_y = min(b['stats'][cv2.CC_STAT_TOP] for b in group)
            max_x = max(b['stats'][cv2.CC_STAT_LEFT] + b['stats'][cv2.CC_STAT_WIDTH] for b in group)
            max_y = max(b['stats'][cv2.CC_STAT_TOP] + b['stats'][cv2.CC_STAT_HEIGHT] for b in group)
            
            combined_w = max_x - min_x
            combined_h = max_y - min_y
            
            # If the door is cut off at top/bottom, the bounding box height is the full image height
            # Combined solidity
            group_solidity = total_area / float(combined_w * combined_h)
            avg_depth = np.mean([b['depth'] for b in group])

            # LOGIC: 
            # An open door (even if split) will have a low combined solidity 
            # because the center (where the robot walks) is empty space.
            if group_solidity < 0.65: # Relaxed slightly for pillars
                door_data.append({
                    'blobs': [b['id'] for b in group],
                    'depth': avg_depth
                })

        # 5. Return the closest valid door group
        filtered_mask = np.zeros_like(door_mask)
        if door_data:
            door_data.sort(key=lambda x: x['depth'])
            for blob_id in door_data[0]['blobs']:
                filtered_mask[labels == blob_id] = 255
                
        return filtered_mask
    
    def get_door_pillars(self, mask, depth_img):
        # 1. Identify all 'door' pixel indices (assuming label is 5)
        v_coords, u_coords = np.where(mask == 255)
        if len(u_coords) < 30: return None

        # 2. Find horizontal extremities
        u_min, u_max = np.min(u_coords), np.max(u_coords)

        # Find corresponding v at those edges to sample depth
        v_min = v_coords[np.argmin(u_coords)]
        v_max = v_coords[np.argmax(u_coords)]

        # Note: These are in camera-space (u, v, depth). 
        return (u_min, v_min, depth_img[v_min, u_min]), (u_max, v_max, depth_img[v_max, u_max])

class DoorLandmarkTracker:
    def __init__(self, dist_threshold=0.8):
        self.registered_doors_ground = []  # List of tuples: (point_left, point_right)
        self.registered_doors_top = []
        self.dist_threshold = dist_threshold
        self.last_world_pos = None
        self.floor_threshold = -1.0

        self.door_detection = DoorDetection()

    def detect_room_transition(self, semantic_image, depth_image, 
                               depth_rotation, depth_position, agent_position):
        """
        Docstring for detect_room_transition
        
        :param semantic_image: semantic image from observation
        :param depth_image: depth image from observation
        :param depth_rotation: depth camera rotation vector with respect to global frame
        :param depth_position: depth camera position vector with respect to global frame
        :param agent_position: agent position vector with respect to global frame
        """
        
        door_pillars = self.detect_door_pillars(semantic_image, depth_image)
        if door_pillars:
            camera_to_world = self.door_detection.get_camera_to_world_coordinate(depth_rotation, depth_position)
            p_left_global = self.door_detection.project_pixel_to_global_frame(*door_pillars[0], camera_to_world)
            p_right_global = self.door_detection.project_pixel_to_global_frame(*door_pillars[1], camera_to_world)

            if self.add_door_landmark(p_left_global, p_right_global, agent_position):
                print("New doorway registered in map.")
        door_idx = self.check_door_crossing(agent_position)

        return door_idx
    
    def detect_door_pillars(self, semantic_image, depth_image):
        """
        Docstring for detect_door
        
        :param semantic_image: semantic image from observation
        :param depth_image: depth image from observation
        """

        door_mask = self.door_detection.get_object_mask(semantic_image)
        depth_image = self.door_detection.decode_depth_image(depth_image)
        result_mask = self.door_detection.filter_doors_with_depth(door_mask, depth_image)
        door_pillars = self.door_detection.get_door_pillars(result_mask, depth_image)

        return door_pillars
    
    def add_door_landmark(self, p1, p2, agent_translation):
        # Convert to numpy for math
        p1, p2 = np.array([p1[0], p1[2]]), np.array([p2[0], p2[2]])  # Use X and Z for ground plane
        midpoint = (p1 + p2) / 2

        # check if door already exists
        if agent_translation[1] < self.floor_threshold:
        # Ground floor
            for existing_p1, existing_p2 in self.registered_doors_ground:
                existing_midpoint = (np.array(existing_p1) + np.array(existing_p2)) / 2
                if np.linalg.norm(midpoint - existing_midpoint) < self.dist_threshold:
                    return False 
            self.registered_doors_ground.append((p1, p2))
        else:
        # Top floor
            for existing_p1, existing_p2 in self.registered_doors_top:
                existing_midpoint = (np.array(existing_p1) + np.array(existing_p2)) / 2
                if np.linalg.norm(midpoint - existing_midpoint) < self.dist_threshold:
                    return False 
            self.registered_doors_top.append((p1, p2))


        return True 
    
    def check_door_crossing(self, current_translation):
        """
        Checks if the robot's movement segment in the world frame 
        intersects any registered door segments.
        
        Args:
            current_translation (np.array): [x, y, z] vector from the agent state.
        Returns:
            int: index of the door crossed, or None
        """
        # Convert to 2D ground plane (assuming Y is up, so we use X and Z)
        # If your 'up' axis is Z, use pos = current_translation[:2]
        current_pos_2d = np.array([current_translation[0], current_translation[2]])

        if self.last_world_pos is None:
            self.last_world_pos = current_pos_2d
            return None

        A = self.last_world_pos
        B = current_pos_2d
        crossed_door_idx = None

        if current_translation[1] < self.floor_threshold:
            for idx, (p1, p2) in enumerate(self.registered_doors_ground):
                # p1 and p2 are world coordinates [x, y, z]
                C = np.array([p1[0], p1[1]])
                D = np.array([p2[0], p2[1]])

                if self._intersect(A, B, C, D):
                    crossed_door_idx = idx
                    break 
        else:
            for idx, (p1, p2) in enumerate(self.registered_doors_top):
                # p1 and p2 are world coordinates [x, y, z]
                C = np.array([p1[0], p1[1]])
                D = np.array([p2[0], p2[1]])

                if self._intersect(A, B, C, D):
                    crossed_door_idx = idx
                    break 

        self.last_world_pos = current_pos_2d
        return crossed_door_idx

    def _intersect(self, A, B, C, D):
        def ccw(p1, p2, p3):
            return (p3[1]-p1[1]) * (p2[0]-p1[0]) > (p2[1]-p1[1]) * (p3[0]-p1[0])
        return (ccw(A, C, D) != ccw(B, C, D)) and (ccw(A, B, C) != ccw(A, B, D))
    
    def reset(self):
        """
        Docstring for reset
        
        :param self: clear variables at the end of the episode
        """
        self.registered_doors_ground = []
        self.registered_doors_top = []
        self.last_world_pos = None

class RoomManager:
    def __init__(self, stairs_id=3):
        self.stairs_id = stairs_id

        # Structure: {"ground": {room_id: data}, "top": {room_id: data}}
        self.floors = {
            "ground": {},
            "top": {}
        }
        self.room_dist_threshold = 1.0  # meters
        self.current_floor = None
        self.current_room_id = None
        self.is_initialized = False

        # Topological Map: door_id -> set of (floor_name, room_id)
        self.connectivity = {}
        
        # Track which door was used to enter each room: (floor, room_id) -> door_id
        self.room_entry_doors = {}
        self.door_tracker = DoorLandmarkTracker()

    def _get_floor_name(self, y_coord):
        """Determine floor based on Y-axis (y down: negative is ground, positive is top)."""
        return "top" if y_coord > -1.0 else "ground"
    
    def _compute_semantic_signature(self, semantic_map):
        """Extract semantic category distribution as room fingerprint."""
        if semantic_map is None:
            return None
        
        if semantic_map.ndim == 3:
            semantic_map = semantic_map[:, :, 0]
        
        # Count unique categories (excluding background/walls)
        unique, counts = np.unique(semantic_map.flatten(), return_counts=True)
        total = counts.sum()
        
        # Create normalized histogram (category -> frequency)
        signature = {}
        for cat, count in zip(unique, counts):
            if cat > 0:  # Exclude background
                signature[int(cat)] = float(count) / total
        
        return signature
    
    def _compare_semantic_signatures(self, sig1, sig2):
        """Compare two semantic signatures using cosine similarity."""
        if sig1 is None or sig2 is None:
            return 0.0
        
        all_keys = set(sig1.keys()) | set(sig2.keys())
        if not all_keys:
            return 0.0
        
        # Compute cosine similarity
        dot_product = sum(sig1.get(k, 0) * sig2.get(k, 0) for k in all_keys)
        norm1 = np.sqrt(sum(v**2 for v in sig1.values()))
        norm2 = np.sqrt(sum(v**2 for v in sig2.values()))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)

    def _initialize_first_room(self, pos, semantic_map=None):
        """Sets up the first room based on the robot's actual starting position."""
        self.current_floor = self._get_floor_name(pos[1])
        self.current_room_id = 0
        self.floors[self.current_floor][0] = {
            "path": [], 
            "doors": set(),
            "semantic_signatures": [],  # List of semantic fingerprints
            "center": None  # Will be computed from path
        }
        if semantic_map is not None:
            sig = self._compute_semantic_signature(semantic_map)
            if sig:
                self.floors[self.current_floor][0]["semantic_signatures"].append(sig)
        self.is_initialized = True
        print(f"INITIALIZED: Started on {self.current_floor} floor in Room 0")
    
    def reset(self):
        """Complete reset after episode ends (clears all data)."""
        print(f"\n=== EPISODE RESET ===")
        self.floors = {"ground": {}, "top": {}}
        self.connectivity = {}
        self.room_entry_doors = {}
        self.current_floor = None
        self.current_room_id = None
        self.is_initialized = False
        self.door_tracker.reset()
        print("  -> All map data cleared")

    def recognize_room_from_path(self, current_pos, floor_name, current_semantic=None, entry_door_id=None):
        """Multi-criteria room matching: topology, geometry, and semantics."""
        candidates = []

        for rid, data in self.floors[floor_name].items():
            path = data["path"]
            if not path:
                continue

            # Criterion 1: Geometric distance to room center/path
            path_arr = np.array(path)
            
            # Use room center if available (more stable than closest path point)
            if data.get("center") is not None:
                center_dist = np.linalg.norm(data["center"] - current_pos)
            else:
                # Fallback to closest path point
                distances = np.linalg.norm(path_arr - current_pos, axis=1)
                center_dist = np.min(distances)
            
            # Criterion 2: Door connectivity check
            # If this room shares a door with our previous room, it's likely adjacent (not the same)
            door_match_score = 0.0
            if entry_door_id is not None and entry_door_id in data["doors"]:
                door_match_score = 1.0  # Strong indicator this could be the room
            
            # Criterion 3: Semantic similarity
            semantic_score = 0.0
            if current_semantic is not None and data.get("semantic_signatures"):
                # Compare with average of stored signatures
                scores = [self._compare_semantic_signatures(current_semantic, sig) 
                         for sig in data["semantic_signatures"]]
                semantic_score = np.mean(scores) if scores else 0.0
            
            # Combined scoring
            # Only consider if within reasonable distance
            if center_dist < self.room_dist_threshold * 2:  # Wider initial filter
                candidates.append({
                    'room_id': rid,
                    'dist': center_dist,
                    'door_match': door_match_score,
                    'semantic': semantic_score,
                    'total_score': door_match_score * 2.0 + semantic_score * 1.5 - center_dist * 0.5
                })
        
        if not candidates:
            return None
        
        # Return best match only if score is strong enough
        candidates.sort(key=lambda x: x['total_score'], reverse=True)
        best = candidates[0]
        
        # Require strong evidence: good door match OR (close distance AND good semantic match)
        if best['door_match'] > 0.5 or (best['dist'] < self.room_dist_threshold and best['semantic'] > 0.6):
            print(f"  -> Matched Room {best['room_id']}: dist={best['dist']:.2f}, door={best['door_match']}, semantic={best['semantic']:.2f}")
            return best['room_id']
        
        return None

    def merge_rooms(self, floor_name, room_to_keep, room_to_remove):
        if room_to_keep == room_to_remove:
            return

        print(f"MERGE: Room {room_to_remove} -> Room {room_to_keep} on {floor_name}")

        target = self.floors[floor_name][room_to_keep]
        source = self.floors[floor_name][room_to_remove]

        target["path"].extend(source["path"])
        target["doors"].update(source["doors"])

        for door_id, rooms_at_door in self.connectivity.items():
            if (floor_name, room_to_remove) in rooms_at_door:
                rooms_at_door.remove((floor_name, room_to_remove))
                rooms_at_door.add((floor_name, room_to_keep))

        del self.floors[floor_name][room_to_remove]

        if self.current_room_id == room_to_remove:
            self.current_room_id = room_to_keep

    def step(self, 
             semantic_image, 
             depth_image, 
             depth_rotation, 
             depth_position, 
             agent_position):
        
        door_idx = self.door_tracker.detect_room_transition(
            semantic_image, depth_image, depth_rotation, depth_position, agent_position)
        self.update(agent_position, semantic_image, door_idx)
        return self.current_room_id

    def update(self, robot_pos, semantic_map=None, door_id=None):
            """
            Updates map with door-centric topology and semantic tracking.
            """
            # 0. Lazy Initialization (fallback for first episode)
            if not self.is_initialized:
                self._initialize_first_room(robot_pos, semantic_map)
                return

            # 1. Check for Stairs
            if semantic_map is not None:
                if semantic_map.ndim == 3:
                    semantic_map = semantic_map[:, :, 0]
                h, w = semantic_map.shape
                center_semantic = semantic_map[h//2, w//2]
                if center_semantic == self.stairs_id:
                    return

            # 2. Handle Floor Change and Door Crossing
            new_floor_name = self._get_floor_name(robot_pos[1])
            old_floor = self.current_floor
            old_room_id = self.current_room_id

            if door_id is not None:
                print(f"\n=== DOOR CROSSING: Door {door_id} from Room {old_room_id} ===")
                
                # STRATEGY 1: Check topology - have we crossed this door from this room before?
                target_info = self._find_room_beyond_door(old_floor, old_room_id, door_id)

                if target_info is not None:
                    # We know exactly where this door leads - use that knowledge!
                    _, self.current_room_id = target_info
                    self.current_floor = new_floor_name
                    print(f"  -> Known door connection: entering Room {self.current_room_id}")
                else:
                    # STRATEGY 2: Unknown door - check if destination matches existing room
                    current_semantic = self._compute_semantic_signature(semantic_map)
                    found_id = self.recognize_room_from_path(
                        robot_pos, new_floor_name, 
                        current_semantic=current_semantic,
                        entry_door_id=door_id
                    )
                    
                    if found_id is None:
                        # Create NEW room
                        all_ids = []
                        for f in self.floors.values():
                            all_ids.extend(f.keys())
                        
                        new_room_id = (max(all_ids) + 1) if all_ids else 0
                        self.floors[new_floor_name][new_room_id] = {
                            "path": [], 
                            "doors": {door_id},
                            "semantic_signatures": [],
                            "center": None
                        }
                        self.current_room_id = new_room_id
                        self.room_entry_doors[(new_floor_name, new_room_id)] = door_id
                        print(f"  -> Created NEW Room {new_room_id}")
                    else:
                        # Found matching existing room
                        self.current_room_id = found_id
                        print(f"  -> Recognized existing Room {found_id}")

                # Update current floor
                self.current_floor = new_floor_name

                # Update Topology (door connects two rooms)
                if door_id not in self.connectivity: 
                    self.connectivity[door_id] = set()
                
                self.connectivity[door_id].add((old_floor, old_room_id))
                self.connectivity[door_id].add((self.current_floor, self.current_room_id))

                # Update door lists
                if old_room_id in self.floors[old_floor]:
                    self.floors[old_floor][old_room_id]["doors"].add(door_id)
                if self.current_room_id in self.floors[self.current_floor]:
                    self.floors[self.current_floor][self.current_room_id]["doors"].add(door_id)
            
            else:
                # No door crossing - just update floor if changed (e.g., stairs)
                self.current_floor = new_floor_name

            # 3. Record Trajectory & Update Room Data
            if self.current_room_id in self.floors[self.current_floor]:
                floor_data = self.floors[self.current_floor]
                room_data = floor_data[self.current_room_id]
                current_path = room_data["path"]
                
                # Add position to path
                if not current_path or np.linalg.norm(np.array(current_path[-1]) - robot_pos) > 0.2:
                    current_path.append(robot_pos.tolist())
                    
                    # Update room center
                    if len(current_path) > 5:
                        path_arr = np.array(current_path)
                        room_data["center"] = np.mean(path_arr, axis=0)
                
                # Add semantic signature periodically
                if semantic_map is not None and len(current_path) % 10 == 0:
                    sig = self._compute_semantic_signature(semantic_map)
                    if sig and len(room_data["semantic_signatures"]) < 20:  # Limit storage
                        room_data["semantic_signatures"].append(sig)
            
            print(f"UPDATED: Floor={self.current_floor}, Room={self.current_room_id}, Doors={self.floors[self.current_floor][self.current_room_id]['doors']}")

    def _check_path_overlap(self, pos, current_rid, floor_name):
        for rid, data in self.floors[floor_name].items():
            if rid == current_rid or not data["path"]: continue
            path_arr = np.array(data["path"])
            dist = np.min(np.linalg.norm(path_arr - pos, axis=1))
            if dist < self.room_dist_threshold: return rid
        return None

    def _find_room_beyond_door(self, floor_name, room_id, door_id):
        if door_id in self.connectivity:
            for f_name, r_id in self.connectivity[door_id]:
                if f_name != floor_name or r_id != room_id:
                    return f_name, r_id
        return None

    def plot_top_down(self, robot_pose=None):
        """Visualizes Ground and Top floor maps side-by-side."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        for ax, f_name in zip([ax1, ax2], ["ground", "top"]):
            ax.set_title(f"{f_name.capitalize()} Floor")
            ax.set_xlabel("X (Forward)")
            ax.set_ylabel("Z (Left)")
            ax.grid(True, linestyle='--', alpha=0.6)

            # Use a color cycle for rooms
            colors = plt.cm.get_cmap('tab10')

            for i, (rid, data) in enumerate(self.floors[f_name].items()):
                if not data["path"]: continue
                path = np.array(data["path"])
                ax.scatter(path[:, 0], path[:, 2], s=10, label=f"Room {rid}", color=colors(i % 10))

            # Mark current robot position if on this floor
            if robot_pose is not None:
                r_pos = robot_pose[:3, 3]
                if self._get_floor_name(r_pos[1]) == f_name:
                    ax.scatter(r_pos[0], r_pos[2], color='red', marker='X', s=100, label="Robot")

            ax.legend(loc='upper right', fontsize='small')

        plt.tight_layout()
        plt.show()