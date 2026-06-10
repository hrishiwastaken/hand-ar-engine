import cv2
import mediapipe as mp
import math
import time
import numpy as np

# --- 3D ENGINE ---
class CubeEngine:
    def __init__(self, center=(150, 150), size=80):
        self.center = list(center)
        self.size = size
        # Base Unit Cube (-1 to 1)
        self.base_points = np.array([
            [-1, -1, -1], [-1, 1, -1], [-1, 1, 1], [-1, -1, 1], # Left Face
            [ 1, -1, -1], [ 1, 1, -1], [ 1, 1, 1], [ 1, -1, 1]  # Right Face
        ])
        self.current_rotation_matrix = np.eye(3)

    def _get_scaled_points(self):
        return self.base_points * (self.size / 2.0)

    def rotate_euler(self, roll, pitch, yaw):
        rx, ry, rz = math.radians(yaw), math.radians(pitch), math.radians(roll)
        rot_x = np.array([[1, 0, 0], [0, math.cos(rx), -math.sin(rx)], [0, math.sin(rx), math.cos(rx)]])
        rot_y = np.array([[math.cos(ry), 0, math.sin(ry)], [0, 1, 0], [-math.sin(ry), 0, math.cos(ry)]])
        rot_z = np.array([[math.cos(rz), -math.sin(rz), 0], [math.sin(rz), math.cos(rz), 0], [0, 0, 1]])
        self.current_rotation_matrix = np.dot(rot_z, np.dot(rot_y, rot_x))
        return np.dot(self._get_scaled_points(), self.current_rotation_matrix.T)

    def align_to_vector(self, p1, p2, up_ref_vector):
        # 1. Update Size (Distance)
        # Using 1:1 normalized Z ensures size remains constant during rotation
        dist = np.linalg.norm(np.array(p1) - np.array(p2))
        if dist > 0: self.size = dist
        
        # 2. Update Center
        self.center[0] = (p1[0] + p2[0]) / 2
        self.center[1] = (p1[1] + p2[1]) / 2

        # 3. Basis Vectors
        vec_x = np.array([p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]])
        norm_x = np.linalg.norm(vec_x)
        if norm_x > 0: vec_x = vec_x / norm_x
        
        vec_up = np.array(up_ref_vector)
        vec_z = np.cross(vec_x, vec_up)
        norm_z = np.linalg.norm(vec_z)
        if norm_z > 0: vec_z = vec_z / norm_z
        else: vec_z = np.array([0, 0, 1])
        
        vec_y = np.cross(vec_z, vec_x)
        norm_y = np.linalg.norm(vec_y)
        if norm_y > 0: vec_y = vec_y / norm_y
        
        R = np.column_stack((vec_x, vec_y, vec_z))
        self.current_rotation_matrix = R
        
        return np.dot(self._get_scaled_points(), R.T)

    def get_frozen_points(self):
        return np.dot(self._get_scaled_points(), self.current_rotation_matrix.T)

    def draw(self, img, points, color=(0, 255, 255), mode="DEFAULT"):
        projected = []
        for p in points:
            x = int(p[0] + self.center[0])
            y = int(p[1] + self.center[1])
            projected.append((x, y))
        
        left_indices = [0, 1, 2, 3]
        right_indices = [4, 5, 6, 7]
        
        for i in range(4):
            cv2.line(img, projected[i], projected[i+4], color, 2, cv2.LINE_AA)
            cv2.line(img, projected[i], projected[(i+1)%4], color, 2, cv2.LINE_AA)
            cv2.line(img, projected[i+4], projected[((i+1)%4)+4], color, 2, cv2.LINE_AA)

        hull_pts = cv2.convexHull(np.array(projected, np.int32))
        overlay = img.copy()
        cv2.fillPoly(overlay, [hull_pts], color)
        cv2.addWeighted(overlay, 0.2, img, 0.8, 0, img)

        if mode == "TACTILE":
            # Highlight Faces
            left_poly = np.array([projected[i] for i in left_indices], np.int32)
            right_poly = np.array([projected[i] for i in right_indices], np.int32)
            
            overlay = img.copy()
            cv2.fillPoly(overlay, [left_poly], (255, 0, 255))
            cv2.fillPoly(overlay, [right_poly], (255, 0, 255))
            cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
            
            # Green Axle
            lx = int(sum([projected[i][0] for i in left_indices]) / 4)
            ly = int(sum([projected[i][1] for i in left_indices]) / 4)
            rx = int(sum([projected[i][0] for i in right_indices]) / 4)
            ry = int(sum([projected[i][1] for i in right_indices]) / 4)
            
            cv2.line(img, (lx, ly), (rx, ry), (0, 255, 0), 4, cv2.LINE_AA)
            cv2.circle(img, (lx, ly), 5, (0, 255, 0), -1)
            cv2.circle(img, (rx, ry), 5, (0, 255, 0), -1)

# --- SMOOTHING ---
class ExponentialSmoother:
    def __init__(self, default_alpha=0.15):
        self.default_alpha = default_alpha
        self.val = 0
        self.initialized = False
    def update(self, new_val, dynamic_rate=None):
        rate = dynamic_rate if dynamic_rate is not None else self.default_alpha
        if not self.initialized:
            self.val = new_val; self.initialized = True
        else:
            if abs(new_val - self.val) > 180: self.val = new_val 
            else: self.val = rate * new_val + (1 - rate) * self.val
        return self.val
    def set(self, val):
        self.val = val; self.initialized = True

class VectorSmoother:
    def __init__(self, alpha=0.3):
        self.x = ExponentialSmoother(alpha)
        self.y = ExponentialSmoother(alpha)
        self.z = ExponentialSmoother(alpha)
    def update(self, v):
        return [self.x.update(v[0]), self.y.update(v[1]), self.z.update(v[2])]
    def set(self, v):
        self.x.set(v[0]); self.y.set(v[1]); self.z.set(v[2])

def get_stability_factor(vector_length, min_safe=20, full_safe=60):
    if vector_length < min_safe: return 0.0
    elif vector_length > full_safe: return 1.0
    else: return (vector_length - min_safe) / (full_safe - min_safe)

def soft_bumper(val, limit=85, softness=10):
    if abs(val) < (limit - softness): return val
    else:
        sign = 1 if val > 0 else -1
        abs_val = abs(val)
        excess = abs_val - (limit - softness)
        compressed = softness + (math.log1p(excess) * 5)
        return sign * min(limit, (limit - softness) + compressed)

def draw_text(img, text, pos, scale=0.6, color=(255, 255, 255), thickness=1):
    x, y = pos
    cv2.putText(img, text, (int(x)+1, int(y)+1), cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), thickness+2, cv2.LINE_AA)
    cv2.putText(img, text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

# ==============================================================================
#                            HAND TRACKING & OCCLUSION
# ==============================================================================
class PreciseHandTracker:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False, max_num_hands=1,
            min_detection_confidence=0.6, min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.tip_ids = [4, 8, 12, 16, 20]

    def find_hands(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)
        return img

    def get_hand_data(self, img):
        if not self.results.multi_hand_landmarks: return 0, None, []
        my_hand = self.results.multi_hand_landmarks[0]
        lm_list = []
        h, w, c = img.shape
        for id, lm in enumerate(my_hand.landmark):
            cx, cy = int(lm.x * w), int(lm.y * h)
            lm_list.append([id, cx, cy, lm.z])
        fingers = []
        right_oriented = lm_list[2][1] < lm_list[17][1]
        fingers.append(1 if (lm_list[4][1] < lm_list[3][1] if right_oriented else lm_list[4][1] > lm_list[3][1]) else 0)
        for id in range(1, 5):
            fingers.append(1 if lm_list[self.tip_ids[id]][2] < lm_list[self.tip_ids[id]-2][2] else 0)
        return fingers.count(1), fingers, lm_list

def apply_hand_occlusion(img_canvas, img_original, lm_list):
    if not lm_list: return img_canvas
    mask = np.zeros(img_canvas.shape[:2], dtype=np.uint8)
    connections = [[0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [0, 17, 18, 19, 20]]
    for chain in connections:
        pts = []
        for id in chain: pts.append((lm_list[id][1], lm_list[id][2]))
        cv2.polylines(mask, [np.array(pts, np.int32)], False, (255), 30, cv2.LINE_AA)
    
    palm_ids = [0, 5, 9, 13, 17]
    palm_pts = np.array([(lm_list[i][1], lm_list[i][2]) for i in palm_ids], np.int32)
    cv2.fillConvexPoly(mask, palm_pts, (255))
    
    mask = cv2.GaussianBlur(mask, (15, 15), 0)
    mask_3ch = cv2.merge([mask, mask, mask])
    hand_pixels = cv2.bitwise_and(img_original, mask_3ch)
    bg_pixels = cv2.bitwise_and(img_canvas, cv2.bitwise_not(mask_3ch))
    return cv2.add(bg_pixels, hand_pixels)

# ==============================================================================
#                               MAIN LOOP
# ==============================================================================
def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    tracker = PreciseHandTracker()
    
    window_name = "AR Final Master"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    cube = CubeEngine(center=(300, 300), size=100)
    
    # Smoothers
    smooth_roll = ExponentialSmoother(0.15)
    smooth_pitch = ExponentialSmoother(0.15)
    smooth_twist = ExponentialSmoother(0.15)
    
    smooth_p1 = VectorSmoother(0.5)
    smooth_p2 = VectorSmoother(0.5)
    smooth_pos_x = ExponentialSmoother(0.25)
    smooth_pos_y = ExponentialSmoother(0.25)
    
    # State Logic
    current_mode = "DISCONNECTED"
    timer_wireless = 0; timer_drag = 0; timer_tactile = 0
    TIMER_THRESHOLD = 30
    
    # Persistent Variables
    tracker_roll = 0; tracker_pitch = 0; tracker_twist = 0
    render_roll = 0; render_pitch = 0; render_twist = 0
    drag_offset_x = 0; drag_offset_y = 0
    locked_finger_ids = None
    
    first_run = True

    while True:
        success, img = cap.read()
        if not success: break
        img = cv2.flip(img, 1)
        h, w, c = img.shape 
        
        img_original = img.copy()

        if first_run:
             cube.center = [int(w * 0.2), int(h * 0.3)]
             first_run = False
        
        img = tracker.find_hands(img)
        count, fingers_list, lm_list = tracker.get_hand_data(img)
        
        finger_pos_map = {}
        wx, wy = 0, 0
        if lm_list:
            wx, wy = lm_list[0][1], lm_list[0][2]
            for item in lm_list:
                # 1:1 Z-Normalization for Physical Logic
                finger_pos_map[item[0]] = [item[1], item[2], item[3] * w * 1.0]

        # Scissor Check
        scissor_angle = 100
        if count == 2 and fingers_list[1] and fingers_list[2] and lm_list:
            ix, iy = lm_list[8][1], lm_list[8][2]
            mx, my = lm_list[12][1], lm_list[12][2]
            v1 = np.array([ix-wx, iy-wy]); v2 = np.array([mx-wx, my-wy])
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 > 0 and n2 > 0:
                cos_a = np.dot(v1, v2) / (n1 * n2)
                scissor_angle = np.degrees(np.arccos(np.clip(cos_a, -1, 1)))

        # --- 1. WIRELESS PHYSICS ---
        if tracker.results.multi_hand_landmarks:
            raw = tracker.results.multi_hand_landmarks[0].landmark
            knuckles = [5, 9, 13, 17]
            vx, vy, vz = 0, 0, 0
            for k in knuckles: vx+=raw[k].x; vy+=raw[k].y; vz+=raw[k].z
            vx, vy, vz = vx/4, vy/4, vz/4

            dx, dy = (vx - raw[0].x)*w, (vy - raw[0].y)*h
            dz = (vz - raw[0].z)*w
            mag_2d = math.hypot(dx, dy)
            pitch_conf = get_stability_factor(mag_2d, 30, 80)

            idx, mid = raw[5], raw[9]; rng, pky = raw[13], raw[17]
            h1x, h1z = (idx.x+mid.x)/2, (idx.z+mid.z)/2
            h2x, h2z = (rng.x+pky.x)/2, (rng.z+pky.z)/2
            h_dx, h_dz = (h2x - h1x)*w, (h2z - h1z)*w * 3.0 # Amplified Z for Wireless Twist Only
            h_mag = math.hypot(h_dx, (h2z-h1z)*h) 
            twist_conf = get_stability_factor(h_mag, 30, 80)

            raw_r = -(math.degrees(math.atan2(dy, dx)) + 90)
            raw_p = soft_bumper(-(math.degrees(math.atan2(dz, mag_2d)) * 1.5))
            raw_t = soft_bumper(math.degrees(math.atan2(h_dz, h_dx)), limit=170, softness=30)

            tracker_roll = smooth_roll.update(raw_r)
            tracker_pitch = smooth_pitch.update(raw_p, dynamic_rate=0.15 * pitch_conf)
            tracker_twist = smooth_twist.update(raw_t, dynamic_rate=0.15 * twist_conf)

        # --- 2. INTERACTION DETECTION ---
        wrist_dist_to_cube = 1000
        active_fingers = [] 
        CONTACT_RADIUS = cube.size * 1.3 
        
        if lm_list:
            wrist_dist_to_cube = math.hypot(wx - cube.center[0], wy - cube.center[1])
            for i, tip_id in enumerate([4, 8, 12, 16, 20]):
                if fingers_list[i] == 1: 
                    tx, ty = lm_list[tip_id][1], lm_list[tip_id][2]
                    d_tip = math.hypot(tx - cube.center[0], ty - cube.center[1])
                    if d_tip > cube.size * 0.5 and d_tip < CONTACT_RADIUS:
                        active_fingers.append((tip_id, tx, ty))

        # --- 3. STATE MACHINE ---
        timer_wireless = max(0, timer_wireless - 1)
        timer_drag = max(0, timer_drag - 1)
        timer_tactile = max(0, timer_tactile - 1)

        if current_mode == "DISCONNECTED":
            if lm_list:
                # A. DRAG
                if count == 0 and wrist_dist_to_cube < CONTACT_RADIUS:
                    timer_drag += 2
                    cv2.circle(img, (wx, wy), 15, (50, 50, 50), -1)
                    cv2.ellipse(img, (wx, wy), (15, 15), 0, 0, int((timer_drag/TIMER_THRESHOLD)*360), (0, 255, 255), 4)
                    if timer_drag > TIMER_THRESHOLD:
                        current_mode = "DRAGGING"
                        drag_offset_x = cube.center[0] - wx
                        drag_offset_y = cube.center[1] - wy
                        smooth_pos_x.set(cube.center[0]); smooth_pos_y.set(cube.center[1])

                # B. WIRELESS
                elif count == 0 and wrist_dist_to_cube >= CONTACT_RADIUS:
                    timer_wireless += 2
                    if timer_wireless > 5:
                        cv2.line(img, (wx, wy), tuple(map(int, cube.center)), (0, 255, 0), 2)
                        mid_x, mid_y = (wx+int(cube.center[0]))//2, (wy+int(cube.center[1]))//2
                        cv2.circle(img, (mid_x, mid_y), 10, (0, 255, 0), -1)
                        fill = int((timer_wireless/TIMER_THRESHOLD)*10)
                        cv2.circle(img, (mid_x, mid_y), fill, (255, 255, 255), -1)
                    if timer_wireless > TIMER_THRESHOLD:
                        current_mode = "WIRELESS"

                # C. TACTILE ROTATE
                elif len(active_fingers) >= 2:
                    best_pair = None; max_dist = 0
                    for i in range(len(active_fingers)):
                        for j in range(i+1, len(active_fingers)):
                            f1 = active_fingers[i]; f2 = active_fingers[j]
                            dist = math.hypot(f1[1]-f2[1], f1[2]-f2[2])
                            if dist > max_dist: max_dist = dist; best_pair = (f1, f2)
                    
                    if best_pair and max_dist > 40:
                        timer_tactile += 2
                        p1_pos = (best_pair[0][1], best_pair[0][2])
                        p2_pos = (best_pair[1][1], best_pair[1][2])
                        cv2.line(img, p1_pos, p2_pos, (255, 255, 0), 3)
                        mid_px, mid_py = (p1_pos[0]+p2_pos[0])//2, (p1_pos[1]+p2_pos[1])//2
                        cv2.ellipse(img, (mid_px, mid_py), (15, 15), 0, 0, int((timer_tactile/TIMER_THRESHOLD)*360), (255, 255, 0), 4)
                        
                        if timer_tactile > TIMER_THRESHOLD:
                            current_mode = "TACTILE_ROTATING"
                            locked_finger_ids = (best_pair[0][0], best_pair[1][0])
                            
                            p1_3d = finger_pos_map[best_pair[0][0]]
                            p2_3d = finger_pos_map[best_pair[1][0]]
                            smooth_p1.set(p1_3d)
                            smooth_p2.set(p2_3d)
                            smooth_pos_x.set(cube.center[0]); smooth_pos_y.set(cube.center[1])

        elif current_mode == "WIRELESS":
            if tracker.results.multi_hand_landmarks:
                render_roll = tracker_roll
                render_pitch = tracker_pitch
                render_twist = tracker_twist
            if count == 2 and scissor_angle < 20:
                current_mode = "DISCONNECTED"
                draw_text(img, "CUT!", (int(cube.center[0]), int(cube.center[1])+100), 2, (0,0,255), 4)
                timer_wireless = 0

        elif current_mode == "DRAGGING":
            if count == 0 and lm_list:
                target_x = wx + drag_offset_x; target_y = wy + drag_offset_y
                cube.center[0] = int(smooth_pos_x.update(target_x))
                cube.center[1] = int(smooth_pos_y.update(target_y))
                draw_text(img, "DRAGGING", (int(cube.center[0])-50, int(cube.center[1])+100), 1, (0, 255, 255), 2)
            else:
                current_mode = "DISCONNECTED"
                timer_drag = 0

        elif current_mode == "TACTILE_ROTATING":
            # 1. FIST RELEASE (The "Panic Button")
            if count == 0:
                current_mode = "DISCONNECTED"
                draw_text(img, "RELEASED", (int(cube.center[0]), int(cube.center[1])+100), 2, (0,0,255), 4)
            else:
                id1, id2 = locked_finger_ids
                if id1 in finger_pos_map and id2 in finger_pos_map:
                    raw_p1 = finger_pos_map[id1]
                    raw_p2 = finger_pos_map[id2]
                    
                    p1 = smooth_p1.update(raw_p1)
                    p2 = smooth_p2.update(raw_p2)
                    
                    # Visuals
                    cv2.line(img, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (255, 255, 0), 3, cv2.LINE_AA)
                    cv2.circle(img, (int(p1[0]), int(p1[1])), 6, (255, 255, 0), -1) 
                    cv2.circle(img, (int(p2[0]), int(p2[1])), 6, (255, 255, 0), -1) 
                    
                    # Update Position
                    curr_mid_x = (p1[0] + p2[0]) / 2
                    curr_mid_y = (p1[1] + p2[1]) / 2
                    cube.center[0] = int(smooth_pos_x.update(curr_mid_x, dynamic_rate=0.4))
                    cube.center[1] = int(smooth_pos_y.update(curr_mid_y, dynamic_rate=0.4))
                    
                    # Rotation (Basis Alignment)
                    spine_vec = [0, -1, 0]
                    if tracker.results.multi_hand_landmarks:
                        mid = lm_list[9]
                        wrist = lm_list[0]
                        spine_vec = [mid[1]-wrist[1], mid[2]-wrist[2], (mid[3]-wrist[3])*w*3.0]
                    
                    final_cube_points = cube.align_to_vector(p1, p2, spine_vec)
                    draw_text(img, "PHYSICAL GRIP", (int(cube.center[0])-50, int(cube.center[1])+100), 1, (255, 255, 0), 2)
                else:
                    current_mode = "DISCONNECTED"
                    timer_tactile = 0

        # --- DRAW ---
        rotated_pts = None
        
        # 1. Determine Points
        if current_mode == "WIRELESS": 
            color = (0, 255, 0)
            rotated_pts = cube.rotate_euler(tracker_roll, tracker_pitch, tracker_twist)
        elif current_mode == "DRAGGING": 
            color = (0, 255, 255)
            rotated_pts = cube.get_frozen_points()
        elif current_mode == "TACTILE_ROTATING": 
            color = (255, 255, 0) # Cyan
            if 'final_cube_points' not in locals(): final_cube_points = cube.get_frozen_points()
            rotated_pts = final_cube_points
        else: 
            color = (0, 0, 255) # Red
            rotated_pts = cube.get_frozen_points()

        # 2. Draw Cube
        mode_str = "TACTILE" if current_mode == "TACTILE_ROTATING" else "DEFAULT"
        cube.draw(img, rotated_pts, color, mode=mode_str)
        
        # 3. Apply Occlusion
        img = apply_hand_occlusion(img, img_original, lm_list)
        
        # 4. Draw Skeleton ON TOP
        if tracker.results.multi_hand_landmarks:
            for hand_lms in tracker.results.multi_hand_landmarks:
                tracker.mp_draw.draw_landmarks(img, hand_lms, tracker.mp_hands.HAND_CONNECTIONS,
                    tracker.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    tracker.mp_draw.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1))

        tx, ty = int(w*0.8), int(h*0.05)
        line_h = 20
        cv2.rectangle(img, (tx-10, ty-20), (tx+220, ty+180), (0,0,0), -1)
        vals = [
            ("MODE", current_mode, color),
            ("SCISSOR", f"{int(scissor_angle)}", (200,200,200)),
            ("TOUCH PTS", f"{len(active_fingers)}", (255,255,255))
        ]
        for label, val, col in vals:
            draw_text(img, f"{label}: {val}", (tx, ty), 0.5, col, 1)
            ty += line_h

        cv2.imshow(window_name, img)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()