import time
import numpy as np
import dxcam
from ultralytics import YOLO
from mouse_instruct import MouseInstruct
import screeninfo
from colorama import Fore
import keyboard, mouse

# -----------------------------
# Configs
# -----------------------------
DEFAULT_CONF = 0.35
SMOOTH_MAX = 1.0
SMOOTH_MIN = 0.3
FLICK_MULT = 1.4
REGION_SCALE = 0.4
TARGET_MIN_SIZE = 20
TARGET_MAX_DIST = 300

class Apex:
    def __init__(self, model_path="./yolov8s.pt", conf=DEFAULT_CONF, port="COM6"):
        monitor = screeninfo.get_monitors()[0]
        self.screen_w, self.screen_h = monitor.width, monitor.height
        rw, rh = int(self.screen_w * REGION_SCALE), int(self.screen_h * REGION_SCALE)
        x1, y1 = (self.screen_w - rw)//2, (self.screen_h - rh)//2
        self.capture_region = (x1, y1, x1+rw, y1+rh)
        self.center_x, self.center_y = rw//2, rh//2

        self.cam = dxcam.create(output_idx=0)
        self.mouse = MouseInstruct.get_mouse(port)
        self.model = YOLO(model_path)
        self.conf = conf
        self.running = False
        print(f"{Fore.LIGHTGREEN_EX}[INFO] Capture region = {self.capture_region}")
        print(f"{Fore.LIGHTGREEN_EX}[INFO] Model loaded: {model_path}")

    def get_xy(self, target_part="head"):
        frame = self.cam.grab(region=self.capture_region)
        if frame is None: return None, None, None
        frame = np.array(frame, dtype=np.uint8)
        results = self.model.predict(frame, device=0, conf=self.conf, verbose=False)

        dx, dy, min_dist = None, None, float('inf')
        if results and len(results) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            for x1, y1, x2, y2 in boxes:
                w, h = x2-x1, y2-y1
                if w < TARGET_MIN_SIZE or h < TARGET_MIN_SIZE: continue
                cx = (x1 + x2)/2
                cy = y1 + (y2 - y1) * (0.25 if target_part=="head" else 0.5)
                tx, ty = cx - self.center_x, cy - self.center_y
                dist = np.hypot(tx, ty)
                if dist > TARGET_MAX_DIST: continue
                if dist < min_dist:
                    dx, dy, min_dist = tx, ty-10, dist
        return dx, dy, min_dist

    def update(self, magnet_key, flick_key, aimbot_var, silent_var, smooth_var, targets, norecoil_var):
        while self.running:
            smooth_factor = max(smooth_var.get()/100, SMOOTH_MIN)
            try:
                def key_pressed(key):
                    key_val = key.lower()
                    if key_val in ["mouse4","mouse5"]: return mouse.is_pressed(key_val)
                    return keyboard.is_pressed(key_val)

                if aimbot_var.get() and magnet_key.get() and targets and key_pressed(magnet_key.get()):
                    dx, dy, dist = self.get_xy(next(iter(targets)))
                    if dx is not None:
                        adaptive_smooth = SMOOTH_MAX if dist>150 else smooth_factor
                        self.mouse.move(int(dx*adaptive_smooth), int(dy*adaptive_smooth))

                if silent_var.get() and flick_key.get() and targets and key_pressed(flick_key.get()):
                    dx, dy, _ = self.get_xy(next(iter(targets)))
                    if dx is not None:
                        self.mouse.silent_flick(int(dx*FLICK_MULT), int(dy*FLICK_MULT))

                if norecoil_var.get():
                    self.mouse.move(0,0)

            except Exception as e:
                print(f"[ERROR] Bot loop: {e}")

            time.sleep(0.008)
