import numpy as np
import cv2
import quaternion

def build_intrinsic_matrix():
    """Return depth-camera intrinsics for ScanNet or HM3D."""

    hfov = 79 * np.pi / 180.0
    fx = 1.0 / np.tan(hfov / 2.0)
    fy = fx * (640.0 / 480.0)
    return {
        "fx": fx,
        "fy": fy,
        "cx": 1.0,
        "cy": 1.0
    }
intrinsics = build_intrinsic_matrix()

def get_world_camera(r, t):
    rotation = quaternion.as_rotation_matrix(r)
    position = t
    world_camera = np.eye(4)
    world_camera[0:3, 0:3] = rotation
    world_camera[0:3, 3] = position
    return world_camera

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

def project_pixel_to_local_frame(u, v, depth, pose):
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

    x_cam = float(x_ndc * z / fx)
    y_cam = float(y_ndc * z / fy)
    z_cam = float(-z)  # right-handed camera, −Z forward
    # 3. Transform to Robot Local Frame (Rotation then Translation)
    # Note: Camera usually has Z-forward, X-right.
    # Robot frames often use X-forward. Adjust based on your setup.
    # Assuming Camera Z axis aligns with Robot X (forward) axis:

    # Rotation matrix for Yaw
    #theta = robot_pose['yaw']
    #cos_t, sin_t = np.cos(theta), np.sin(theta)

    # Rotate camera coordinates (x_cam, z_cam) to align with world frame
    # (Simple 2D rotation for XY plane assuming Z_cam is depth)
    #x_world = robot_pose['x'] + (z_cam * cos_t - x_cam * sin_t)
    #y_world = robot_pose['y'] + (z_cam * sin_t + x_cam * cos_t)
    world_h = pose @ np.array([x_cam, y_cam, z_cam, 1.0], dtype=np.float32)
    return world_h[:3] / world_h[3]

def filter_doors_with_depth(door_mask, depth_map, depth_threshold=4.0):
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
        self.registered_doors_ground = []  # List of tuples: (point_left, point_right)
        self.registered_doors_top = []
        self.dist_threshold = dist_threshold
        self.last_world_pos = None
        self.floor_threshold = -1.0

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

    def add_door_landmark(self, p1, p2, agent_translation):
        # Convert to numpy for math
        p1, p2 = np.array([p1[0], p1[2]]), np.array([p2[0], p2[2]])  # Use X and Z for ground plane
        midpoint = (p1 + p2) / 2

        if agent_translation[1] < self.floor_threshold:
        # Check if this door already exists in memory
            for existing_p1, existing_p2 in self.registered_doors_ground:
                existing_midpoint = (np.array(existing_p1) + np.array(existing_p2)) / 2
                if np.linalg.norm(midpoint - existing_midpoint) < self.dist_threshold:
                    return False 
            self.registered_doors_ground.append((p1, p2))
        else:
        # Check if this door already exists in memory
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
        self.registered_doors_ground = []
        self.registered_doors_top = []
        self.last_world_pos = None
