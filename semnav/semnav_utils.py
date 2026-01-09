import numpy as np
import cv2
from sklearn.cluster import DBSCAN
import time

class UnifiedRoomTriggerSystem:
    def __init__(self, association_dist=1.0, cooldown_sec=5.0):
        """
        association_dist: Max distance (meters) to match a detection to an existing door.
        cooldown_sec: Minimum time between VLM triggers for the same/nearby door.
        """
        # Global Map of Doorways: { door_id: {'p1': [x,y], 'p2': [x,y], 'midpoint': [x,y]} }
        self.global_doors = {}
        self.next_id = 0

        # Logic state
        self.association_threshold = association_dist
        self.cooldown_period = cooldown_sec
        self.last_trigger_time = 0
        self.last_crossed_id = None

    def process_frame(self, semantic_mask, depth_img, intrinsics, robot_pose, robot_traj):
        """
        Main entry point for every frame.
        Returns: (bool trigger_vlm, int door_id)
        """
        # 1. Project 'door' pixels to 3D and separate clusters that are merged in 2D
        clusters = self._separate_doors_3d(semantic_mask, depth_img, intrinsics)

        # 2. Convert clusters to global landmarks and associate with memory
        self._update_landmarks(clusters, robot_pose)

        # 3. Use robot trajectory to check for line-segment intersection
        door_crossed = self._check_crossing_event(robot_traj)

        if door_crossed is not None:
            current_time = time.time()
            # Cooldown check: prevent rapid-fire triggers if robot lingers in doorway
            if (current_time - self.last_trigger_time > self.cooldown_period) or (door_crossed != self.last_crossed_id):
                self.last_trigger_time = current_time
                self.last_crossed_id = door_crossed
                return True, door_crossed

        return False, None

    def _separate_doors_3d(self, mask, depth, intrinsics, door_label=255):
        # Filter for door pixels
        v_idx, u_idx = np.where(mask == door_label)
        if len(u_idx) < 50: return [] # Ignore tiny noise clusters

        # Project pixels into Camera Frame coordinates (X, Y, Z)
        z = depth[v_idx, u_idx]
        x = (u_idx - intrinsics['cx']) * z / intrinsics['fx']
        y = (v_idx - intrinsics['cy']) * z / intrinsics['fy']
        points_3d = np.stack((x, y, z), axis=-1)

        # DBSCAN: Clusters points based on actual physical 3D distance
        # eps 0.25: points within 25cm belong to the same door.
        clustering = DBSCAN(eps=0.25, min_samples=30).fit(points_3d)

        # Return a list of point-clouds, one for each separated door
        return [points_3d[clustering.labels_ == i] for i in set(clustering.labels_) if i != -1]

    def _update_landmarks(self, clusters, pose):
        for cluster in clusters:
            # Extract pillars: Leftmost and Rightmost points in the cluster
            p1_cam = cluster[np.argmin(cluster[:, 0])]
            p2_cam = cluster[np.argmax(cluster[:, 0])]

            # Transform from camera-view to global map coordinates
            p1_global = self._to_global(p1_cam, pose)
            p2_global = self._to_global(p2_cam, pose)
            midpoint = (p1_global + p2_global) / 2

            # Data Association: Is this a door we've seen before?
            matched_id = None
            for d_id, data in self.global_doors.items():
                if np.linalg.norm(midpoint - data['midpoint']) < self.association_threshold:
                    matched_id = d_id
                    break

            if matched_id is not None:
                # Smoothing: slightly adjust the existing pillars with new data
                alpha = 0.3
                self.global_doors[matched_id]['p1'] = (1-alpha) * self.global_doors[matched_id]['p1'] + alpha * p1_global
                self.global_doors[matched_id]['p2'] = (1-alpha) * self.global_doors[matched_id]['p2'] + alpha * p2_global
                self.global_doors[matched_id]['midpoint'] = (self.global_doors[matched_id]['p1'] + self.global_doors[matched_id]['p2']) / 2
            else:
                # Register new unique door
                self.global_doors[self.next_id] = {'p1': p1_global, 'p2': p2_global, 'midpoint': midpoint}
                self.next_id += 1

    def _to_global(self, p_cam, pose):
        # Rotation (Yaw) + Translation from Robot Frame to Global Frame
        theta = pose['yaw']
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        # p_cam[2] is depth (X forward for robot), p_cam[0] is lateral (Y left/right)
        gx = pose['x'] + (p_cam[2] * cos_t - p_cam[0] * sin_t)
        gy = pose['y'] + (p_cam[2] * sin_t + p_cam[0] * cos_t)
        return np.array([gx, gy])

    def _check_crossing_event(self, traj):
        if len(traj) < 2: return None
        r_prev, r_curr = np.array(traj[-2]), np.array(traj[-1])

        for d_id, data in self.global_doors.items():
            # Trigger if robot path intersects the door pillar segment
            if self._intersect(data['p1'], data['p2'], r_prev, r_curr):
                return d_id
        return None

    def _intersect(self, A, B, C, D):
        # Standard Counter-Clockwise (CCW) check for line segment intersection
        def ccw(p1, p2, p3):
            return (p3[1]-p1[1]) * (p2[0]-p1[0]) > (p2[1]-p1[1]) * (p1[0]-p1[0])
        return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def get_object_mask(img_array):
    """
    Loads an image and returns a binary mask where the pixels matching 
    the hex_color are 1 (True) and others are 0 (False).
    """
    #hex_code = '315E38'  # Door color in hex

    # Convert target hex to RGB array
    #target_rgb = np.array(tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4)))

    # Compare each pixel to the target RGB values
    # np.all checks if R, G, and B all match simultaneously across the last axis
    mask = np.all(img_array == 8, axis=-1)

    return mask.astype(np.uint8) * 255 

def project_pixel_to_local_frame(u, v, depth, intrinsics, robot_pose):
    """
    u, v: Pixel coordinates of the door pillar
    depth: Depth value at (u, v) from depth image
    intrinsics: Dictionary with 'fx', 'fy', 'cx', 'cy'
    robot_pose: Dictionary with 'x', 'y', 'yaw' (in radians)
    """
    height = 480
    width = 640
    # 1. Camera Intrinsics
    fx, fy = intrinsics['fx'], intrinsics['fy']
    cx, cy = intrinsics['cx'], intrinsics['cy']

    # 2. Project to 3D Camera Frame (Standard Pinhole Model)
    # Z is depth, X and Y are calculated based on similarity
    #z_cam = depth
    #x_cam = (u - cx) * z_cam / fx
    #y_cam = (v - cy) * z_cam / fy
    z = depth
    x_ndc = (u / (width - 1)) * 2.0 - 1.0
    y_ndc = 1.0 - (v / (height - 1)) * 2.0

    x_cam = x_ndc * z / fx
    y_cam = y_ndc * z / fy
    z_cam = -z  # right-handed camera, −Z forward
    # 3. Transform to Robot Local Frame (Rotation then Translation)
    # Note: Camera usually has Z-forward, X-right.
    # Robot frames often use X-forward. Adjust based on your setup.
    # Assuming Camera Z axis aligns with Robot X (forward) axis:

    # Rotation matrix for Yaw
    theta = robot_pose['yaw']
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    # Rotate camera coordinates (x_cam, z_cam) to align with world frame
    # (Simple 2D rotation for XY plane assuming Z_cam is depth)
    x_world = robot_pose['x'] + (z_cam * cos_t - x_cam * sin_t)
    y_world = robot_pose['y'] + (z_cam * sin_t + x_cam * cos_t)

    return x_world, y_world

def decode_depth_image(depth_normalized, min_depth, max_depth):
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
    actual_depth = min_depth + (depth_float * (max_depth - min_depth))
    
    return actual_depth

def filter_doors_with_depth(door_mask, depth_map, depth_threshold=4.0):
    """
    Advanced filter that handles 'cut-off' door frames by grouping components
    with similar depths and horizontal proximity.
    """
    # 1. Clean up noise
    kernel = np.ones((5,5), np.uint8)
    door_mask = cv2.morphologyEx(door_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)

    # 2. Get connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(door_mask, connectivity=8)
    
    blobs = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 10000: continue # Ignore tiny noise
        
        # Calculate median depth for this specific blob
        blob_depths = depth_map[labels == i]
        median_depth = np.median(blob_depths)
        
        if depth_threshold and median_depth > depth_threshold:
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

class DoorLandmarkTracker:
    def __init__(self, dist_threshold=0.8):
        self.registered_doors = []  # List of tuples: (point_left, point_right)
        self.dist_threshold = dist_threshold

    def get_door_pillars(self, semantic_mask, depth_img):
        # 1. Identify all 'door' pixel indices (assuming label is 5)
        v_coords, u_coords = np.where(semantic_mask == 255)
        if len(u_coords) < 30: return None

        # 2. Find horizontal extremities
        u_min, u_max = np.min(u_coords), np.max(u_coords)

        # Find corresponding v at those edges to sample depth
        v_min = v_coords[np.argmin(u_coords)]
        v_max = v_coords[np.argmax(u_coords)]

        # Note: These are in camera-space (u, v, depth). 
        # For better crossing detection, convert these to World (X, Y) 
        # using your camera intrinsics and robot pose.
        return (u_min, v_min, depth_img[v_min, u_min]), (u_max, v_max, depth_img[v_max, u_max])

    def add_door_landmark(self, p1, p2):
        # Convert to numpy for math
        p1, p2 = np.array(p1), np.array(p2)
        midpoint = (p1 + p2) / 2

        # Check if this door already exists in memory
        for existing_p1, existing_p2 in self.registered_doors:
            existing_midpoint = (np.array(existing_p1) + np.array(existing_p2)) / 2
            if np.linalg.norm(midpoint - existing_midpoint) < self.dist_threshold:
                return False 

        self.registered_doors.append((p1, p2))
        return True 

    def check_door_crossing(self, traj):
        """
        Checks if the robot's path from last_robot_pose to current_robot_pose 
        intersects any registered door.
        
        Args:
            current_robot_pose: Tuple or list (x, y)
        Returns:
            int: index of the door crossed, or None
        """

        if len(traj) < 2: return None
        A, B = np.array(traj[-2]), np.array(traj[-1])

        crossed_door_idx = None

        for idx, (p1, p2) in enumerate(self.registered_doors):
            # C, D are the door pillars. 
            # We only use X and Y (indices 0 and 1) for the intersection check.
            C = (p1[0], p1[1])
            D = (p2[0], p2[1])

            if self._intersect(A, B, C, D):
                crossed_door_idx = idx
                break # Return the first door found

        # Update pose for next check
        return crossed_door_idx

    def _intersect(self, A, B, C, D):
        """
        Standard Counter-Clockwise (CCW) check for line segment intersection.
        """
        def ccw(p1, p2, p3):
            # Returns True if points are in counter-clockwise order
            return (p3[1]-p1[1]) * (p2[0]-p1[0]) > (p2[1]-p1[1]) * (p3[0]-p1[0])
        
        # Two segments AB and CD intersect iff:
        # 1. A and B are on opposite sides of line CD
        # 2. C and D are on opposite sides of line AB
        return (ccw(A, C, D) != ccw(B, C, D)) and (ccw(A, B, C) != ccw(A, B, D))
    
    def reset(self):
        self.registered_doors = []
