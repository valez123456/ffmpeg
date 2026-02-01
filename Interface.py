import os
import time
import threading
import tkinter as tk
from tkinter import ttk
import numpy as np
import dxcam
import keyboard
import mouse
from colorama import Fore
from ultralytics import YOLO
from mouse_instruct import MouseInstruct
import screeninfo

# -----------------------------
# Configs
# -----------------------------
DEFAULT_CONF = 0.45
DEFAULT_SMOOTH = 1
FLICK_MULT = 1.4
REGION_SCALE = 0.5

# -----------------------------
# Utilitaire touche
# -----------------------------
def get_key_value(key_name):
    key_lower = key_name.lower()
    if key_lower in ["mouse4", "mouse5"]:
        return key_lower
    elif key_lower in ["ctrl", "left ctrl", "control gauche"]:
        return "left ctrl"
    return key_lower

# -----------------------------
# Apex Bot
# -----------------------------
class Apex:
    def __init__(self, model_path="./best_8s.pt", conf=DEFAULT_CONF, port="COM4"):
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
        if frame is None:
            return None, None
        frame = np.array(frame, dtype=np.uint8)
        results = self.model.predict(frame, device=0, conf=self.conf, verbose=False)

        dx, dy = None, None
        if results and len(results) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            min_dist = float('inf')
            for x1, y1, x2, y2 in boxes:
                cx = (x1 + x2) / 2
                cy = y1 + (y2 - y1) * (0.25 if target_part == "head" else 0.5)
                tx, ty = cx - self.center_x, cy - self.center_y
                dist = np.hypot(tx, ty)
                if dist < min_dist:
                    min_dist = dist
                    dx, dy = tx, ty-10
        return dx, dy

    def update(self, magnet_key, flick_key, aimbot_var, silent_var, smooth_var, targets, norecoil_var):
        while self.running:
            smooth_factor = max(smooth_var.get() / 100, 0.01)
            try:
                def key_pressed(key):
                    key_val = get_key_value(key)
                    if key_val in ["mouse4", "mouse5"]:
                        return mouse.is_pressed(key_val)
                    return keyboard.is_pressed(key_val)

                # Magnet aimbot
                if aimbot_var.get() and magnet_key.get() and targets:
                    if key_pressed(magnet_key.get()):
                        for target in targets:
                            dx, dy = self.get_xy(target)
                            if dx is not None:
                                self.mouse.move(int(dx * smooth_factor), int(dy * smooth_factor))

                # Silent flick
                if silent_var.get() and flick_key.get() and targets:
                    if key_pressed(flick_key.get()):
                        for target in targets:
                            dx, dy = self.get_xy(target)
                            if dx is not None:
                                self.mouse.silent_flick(int(dx * FLICK_MULT), int(dy * FLICK_MULT))

                # No recoil
                if norecoil_var.get():
                    self.mouse.move(0, 0)

            except Exception as e:
                print(f"[ERROR] Bot loop: {e}")

            time.sleep(0.01)

# -----------------------------
# GUI
# -----------------------------
class ApexGUI:
    def __init__(self, root, port="COM4"):
        self.root = root
        root.title("Apex Bot")
        root.geometry("500x350+50+50")
        root.overrideredirect(True)
        root.configure(bg="black")
        root.withdraw()

        self.apex = Apex(port=port)
        self.bot_running = False
        self.menu_visible = False

        # Variables
        self.magnet_key = tk.StringVar(value="v")
        self.flick_key = tk.StringVar(value="c")
        self.aimbot_var = tk.BooleanVar(value=False)
        self.silent_var = tk.BooleanVar(value=False)
        self.norecoil_var = tk.BooleanVar(value=False)
        self.smooth_var = tk.DoubleVar(value=DEFAULT_SMOOTH * 100)
        self.targets = set()
        self.current_section = "Aimbot"

        self.build_gui()
        self.start_bot()
        keyboard.add_hotkey("insert", self.toggle_menu)

    def toggle_menu(self):
        self.menu_visible = not self.menu_visible
        if self.menu_visible:
            self.root.deiconify()
            self.root.attributes("-topmost", True)
            self.root.focus_force()
            self.root.grab_set()
        else:
            self.root.withdraw()
            self.root.grab_release()

    def build_gui(self):
        frame = tk.Frame(self.root, bg="black")
        frame.pack(fill="both", expand=True)

        # Top bar
        top_bar = tk.Frame(frame, bg="black")
        top_bar.pack(fill="x", pady=2)
        tk.Label(top_bar, text="Apex Bot", font=("Consolas",12,"bold"), fg="white", bg="black").pack(side="left", padx=5)
        tk.Button(top_bar, text="X", command=self.close, font=("Consolas",12,"bold"), fg="white", bg="black", activebackground="red", bd=0).pack(side="right", padx=2)
        top_bar.bind("<Button-1>", self.start_move)
        top_bar.bind("<ButtonRelease-1>", self.stop_move)
        top_bar.bind("<B1-Motion>", self.on_move)

        # Menu gauche
        self.menu_frame = tk.Frame(frame, bg="gray10")
        self.menu_frame.place(x=0, y=30, width=125, height=316)
        self.sections = ["Aimbot","TriggerBot","NoRecoil","ESP","Information"]
        for sec in self.sections:
            tk.Button(self.menu_frame, text=sec, font=("Consolas",10), fg="white", bg="gray10", activebackground="gray30", bd=0,
                      command=lambda s=sec: self.switch_section(s)).pack(fill="x", pady=5)

        self.separator = tk.Frame(frame, bg="white", width=2, height=316)
        self.separator.place(x=125, y=30)
        self.panel_frame = tk.Frame(frame, bg="black")
        self.panel_frame.place(x=130, y=30, width=366, height=316)

        # Crée tous les checkboxes à l’avance
        self.canvas_aimbot, _ = self.create_checkbox(self.panel_frame, "Magnet", self.aimbot_var)
        self.canvas_silent, _ = self.create_checkbox(self.panel_frame, "Silent Flick", self.silent_var)
        self.canvas_norecoil, _ = self.create_checkbox(self.panel_frame, "No Recoil", self.norecoil_var)

        # Smooth slider
        self.slider = tk.Scale(self.panel_frame, from_=0, to=100, orient="horizontal", variable=self.smooth_var, label="Smooth", bg="black", fg="white")

        # Target selection
        self.target_frame = tk.Frame(self.panel_frame, bg="black")
        tk.Label(self.target_frame, text="Target:", fg="white", bg="black", font=("Consolas",12)).pack(side="left")
        self.create_target_checkbox("head")
        self.create_target_checkbox("body")

        # Choix touches Magnet/Silent (clavier + souris)
        options = ["v", "c", "x", "mouse4", "mouse5", "left ctrl"]
        tk.Label(self.panel_frame, text="Magnet Key:", fg="white", bg="black").pack()
        ttk.Combobox(self.panel_frame, textvariable=self.magnet_key, values=options).pack()
        tk.Label(self.panel_frame, text="Silent Key:", fg="white", bg="black").pack()
        ttk.Combobox(self.panel_frame, textvariable=self.flick_key, values=options).pack()

        self.update_interface()

    def create_checkbox(self, master, text, var):
        frame = tk.Frame(master, bg="black")
        frame.pack(fill="x", pady=5)
        canvas = tk.Canvas(frame, width=25, height=25, bg="black", bd=0, highlightthickness=0)
        canvas.pack(side="left", padx=5)
        tk.Label(frame, text=text, fg="white", bg="black", font=("Consolas",12)).pack(side="left")

        def toggle():
            var.set(not var.get())
            redraw()

        def redraw():
            canvas.delete("all")
            canvas.create_rectangle(0, 0, 25, 25, fill="blue" if var.get() else "#555555",
                                    outline="blue" if var.get() else "#555555")

        canvas.bind("<Button-1>", lambda e: toggle())
        canvas.redraw = redraw
        redraw()
        return canvas, var

    def create_target_checkbox(self, name):
        frame = tk.Frame(self.target_frame, bg="black")
        frame.pack(side="left", padx=5)
        canvas = tk.Canvas(frame, width=10, height=10, bg="black", bd=0, highlightthickness=0)
        canvas.pack(side="left")
        tk.Label(frame, text=name.capitalize(), fg="white", bg="black", font=("Consolas",8)).pack(side="left")

        def toggle():
            if name in self.targets:
                self.targets.remove(name)
            else:
                self.targets.add(name)
            redraw()

        def redraw():
            canvas.delete("all")
            canvas.create_rectangle(0, 0, 10, 10, fill="blue" if name in self.targets else "#555555",
                                    outline="blue" if name in self.targets else "#555555")

        canvas.bind("<Button-1>", lambda e: toggle())
        canvas.redraw = redraw
        redraw()

    def start_move(self, event): self.root.x = event.x; self.root.y = event.y
    def stop_move(self, event): self.root.x = None; self.root.y = None
    def on_move(self, event): self.root.geometry(f"+{self.root.winfo_x() + (event.x - self.root.x)}+{self.root.winfo_y() + (event.y - self.root.y)}")

    def switch_section(self, section):
        self.current_section = section
        self.update_interface()

    def update_interface(self):
        for w in self.panel_frame.winfo_children():
            w.pack_forget()
        if self.current_section == "Aimbot":
            self.canvas_aimbot.master.pack(fill="x", pady=5)
            self.canvas_silent.master.pack(fill="x", pady=5)
            self.slider.pack(fill="x", pady=10)
            self.target_frame.pack(fill="x", pady=5)
        elif self.current_section == "NoRecoil":
            self.canvas_norecoil.master.pack(fill="x", pady=5)
        elif self.current_section == "Information":
            tk.Label(self.panel_frame, text="Created by Valez", fg="white", bg="black", font=("Consolas",12)).pack(pady=10)
        else:
            tk.Label(self.panel_frame, text=f"Section: {self.current_section}", fg="white", bg="black", font=("Consolas",12)).pack(pady=10)

    def start_bot(self):
        if not self.bot_running:
            self.apex.running = True
            self.bot_running = True
            threading.Thread(target=self.apex.update, args=(
                self.magnet_key, self.flick_key, self.aimbot_var, self.silent_var, self.smooth_var, self.targets, self.norecoil_var
            ), daemon=True).start()

    def close(self):
        self.apex.running = False
        self.apex.mouse.close()
        self.root.destroy()
        os._exit(0)

# -----------------------------
# Launch
# -----------------------------
if __name__ == "__main__":
    root = tk.Tk()
    gui = ApexGUI(root)
    root.mainloop()
