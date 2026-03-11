"""
Hand Swing Audio Trigger  v4
=============================
Raise BOTH hands and swing in ANY direction to play 67.mp3:
  ↔  Left  / Right
  ↕  Up    / Down
  ⬡  Closer / Further (Z depth)

INSTALL:  pip install mediapipe opencv-python pygame pillow
RUN:      python hand_swing_audio.py
"""

import tkinter as tk
from tkinter import font as tkfont
from PIL import Image, ImageTk
import threading, time, os, sys
from collections import deque

import cv2
import mediapipe as mp
import numpy as np

# ─────────────────────────────────────── CONFIG ────────────────
AUDIO_FILE      = "67.mp3"
SWINGS_NEEDED   = 2      # half-swings on ANY axis needed to trigger
SWING_DELTA_XY  = 0.035  # min travel for X or Y (fraction of frame)
SWING_DELTA_Z   = 0.03   # min travel for Z (mediapipe Z units)
RAISE_THRESHOLD = 0.72   # wrist Y < this = "raised" (very lenient)
COOLDOWN_SEC    = 2.0
CAM_W, CAM_H    = 640, 480

# ───────────────────────────────────────── AUDIO ───────────────
def make_player():
    if not os.path.exists(AUDIO_FILE):
        print(f"[Audio] WARNING: '{AUDIO_FILE}' not found in {os.getcwd()}")
        return None
    try:
        import pygame
        pygame.mixer.pre_init(44100, -16, 2, 512)
        pygame.mixer.init()
        pygame.mixer.music.load(AUDIO_FILE)
        def play():
            pygame.mixer.music.stop()
            pygame.mixer.music.load(AUDIO_FILE)
            pygame.mixer.music.play()
        print("[Audio] pygame OK")
        return play
    except Exception as e:
        print(f"[Audio] pygame failed: {e}")
    try:
        from playsound import playsound
        def play(): playsound(AUDIO_FILE, block=False)
        print("[Audio] playsound OK")
        return play
    except Exception as e:
        print(f"[Audio] playsound failed: {e}")
    if sys.platform == "win32":
        def play():
            import subprocess
            subprocess.Popen(["start","",os.path.abspath(AUDIO_FILE)], shell=True)
        print("[Audio] Windows start OK")
        return play
    print("[Audio] No backend found!  pip install pygame")
    return None


# ──────────────────────────────────── PER-AXIS SWING COUNTER ───
class AxisSwing:
    """Counts direction reversals on a single 1-D signal."""
    def __init__(self, delta):
        self.delta    = delta
        self.buf      = deque(maxlen=30)
        self.count    = 0
        self.last_val = None
        self.direction= None   # +1 or -1

    def reset(self):
        self.buf.clear()
        self.count     = 0
        self.last_val  = None
        self.direction = None

    def update(self, v: float) -> int:
        self.buf.append(v)
        if len(self.buf) < 3:
            return self.count

        cur = float(np.mean(list(self.buf)[-3:]))
        if self.last_val is None:
            self.last_val = cur
            return self.count

        dv      = cur - self.last_val
        if abs(dv) < 0.002:
            return self.count

        new_dir = 1 if dv > 0 else -1
        if self.direction is None:
            self.direction = new_dir
            self.last_val  = cur
            return self.count

        if new_dir != self.direction:
            travel = abs(cur - self.last_val)
            if travel >= self.delta:
                self.count    += 1
                self.direction = new_dir
                self.last_val  = cur
        return self.count


class SwingDetector:
    """Combines X, Y, Z axes — any axis can contribute to the count."""
    def __init__(self):
        self.x = AxisSwing(SWING_DELTA_XY)
        self.y = AxisSwing(SWING_DELTA_XY)
        self.z = AxisSwing(SWING_DELTA_Z)

    def reset(self):
        self.x.reset(); self.y.reset(); self.z.reset()

    def update(self, xv, yv, zv):
        cx = self.x.update(xv)
        cy = self.y.update(yv)
        cz = self.z.update(zv)
        # total = max on any single axis (don't double-count same swing)
        total = max(cx, cy, cz)
        return total, cx, cy, cz


# ──────────────────────────────── CAMERA + MEDIAPIPE THREAD ────
class HandTracker(threading.Thread):
    def __init__(self, frame_cb, state_cb):
        super().__init__(daemon=True)
        self.frame_cb  = frame_cb
        self.state_cb  = state_cb
        self._stop     = threading.Event()
        self.play_fn   = make_player()
        self.detector  = SwingDetector()
        self.last_trig = 0

    def stop(self): self._stop.set()

    def run(self):
        mp_h    = mp.solutions.hands
        mp_draw = mp.solutions.drawing_utils
        hands   = mp_h.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
        if not cap.isOpened():
            self.state_cb({"error": "Cannot open webcam"}); return

        while not self._stop.is_set():
            ok, frame = cap.read()
            if not ok: continue

            frame = cv2.flip(frame, 1)
            h, w  = frame.shape[:2]
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res   = hands.process(rgb)

            state = dict(hands_detected=0, both_raised=False,
                         total=0, cx=0, cy=0, cz=0,
                         triggered=False,
                         cooldown=max(0., COOLDOWN_SEC-(time.time()-self.last_trig)))

            # raise-threshold guide line
            ty = int(RAISE_THRESHOLD * h)
            cv2.line(frame, (0, ty), (w, ty), (80, 80, 220), 1)
            cv2.putText(frame, "raise hands above here",
                        (8, ty-6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120,120,255), 1)

            if res.multi_hand_landmarks:
                state["hands_detected"] = len(res.multi_hand_landmarks)
                wxs, wys, wzs = [], [], []

                for hand_lm in res.multi_hand_landmarks:
                    # draw skeleton
                    mp_draw.draw_landmarks(
                        frame, hand_lm, mp_h.HAND_CONNECTIONS,
                        mp_draw.DrawingSpec(color=(0,255,120), thickness=2, circle_radius=4),
                        mp_draw.DrawingSpec(color=(0,180,80),  thickness=2),
                    )
                    lm = hand_lm.landmark[mp_h.HandLandmark.WRIST]
                    wxs.append(lm.x); wys.append(lm.y); wzs.append(lm.z)
                    # wrist dot
                    cv2.circle(frame, (int(lm.x*w), int(lm.y*h)), 10, (0,255,200), -1)

                both_raised = len(wys)==2 and all(y < RAISE_THRESHOLD for y in wys)
                state["both_raised"] = both_raised

                if both_raised:
                    avg_x = float(np.mean(wxs))
                    avg_y = float(np.mean(wys))
                    avg_z = float(np.mean(wzs))
                    total, cx, cy, cz = self.detector.update(avg_x, avg_y, avg_z)
                    state.update(total=total, cx=cx, cy=cy, cz=cz)

                    # which axis is currently active label
                    axis_label = ""
                    if cx >= cy and cx >= cz: axis_label = f"← → x{cx}"
                    elif cy >= cx and cy >= cz: axis_label = f"↑ ↓ x{cy}"
                    else:                        axis_label = f"⬡ Z x{cz}"

                    cv2.putText(frame, f"SWING!  {axis_label}",
                                (15, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,100), 2)

                    # progress bar
                    ratio = min(total / SWINGS_NEEDED, 1.0)
                    cv2.rectangle(frame, (0,h-20),(w,h),(25,25,25),-1)
                    cv2.rectangle(frame, (0,h-20),(int(ratio*w),h),(0,210,100),-1)
                    cv2.putText(frame, f"Swings: {total}/{SWINGS_NEEDED}  [X:{cx} Y:{cy} Z:{cz}]",
                                (8,h-5), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255,255,255), 1)

                    if total >= SWINGS_NEEDED and (time.time()-self.last_trig) > COOLDOWN_SEC:
                        state["triggered"] = True
                        self.last_trig     = time.time()
                        self.detector.reset()
                        if self.play_fn:
                            threading.Thread(target=self.play_fn, daemon=True).start()
                        print(f"[Gesture] Triggered → {AUDIO_FILE}")
                        ov = frame.copy()
                        cv2.rectangle(ov,(0,0),(w,h),(0,255,100),-1)
                        cv2.addWeighted(ov,0.3,frame,0.7,0,frame)
                        cv2.putText(frame,"PLAYING 67.mp3!",
                                    (w//2-170,h//2),cv2.FONT_HERSHEY_SIMPLEX,
                                    1.3,(255,255,255),3)
                else:
                    self.detector.reset()
                    cv2.putText(frame,
                        "Raise BOTH hands above the line!" if state["hands_detected"]>0 else "No hands — show hands to camera",
                        (15,36), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                        (50,150,255) if state["hands_detected"]>0 else (80,80,80), 2)
            else:
                self.detector.reset()
                cv2.putText(frame,"No hands detected",(15,36),
                            cv2.FONT_HERSHEY_SIMPLEX,0.75,(80,80,80),1)

            self.frame_cb(frame)
            self.state_cb(state)

        cap.release(); hands.close()


# ──────────────────────────────────────── TKINTER APP ──────────
class App(tk.Tk):
    PANEL_H = 120

    def __init__(self):
        super().__init__()
        self.title("Hand Swing → 67.mp3")
        self.configure(bg="#0d0d1a")
        self.resizable(False, False)
        self.protocol("WM_DELETE_WINDOW", self._quit)
        self.geometry(f"{CAM_W}x{CAM_H+self.PANEL_H}")

        f_big = tkfont.Font(family="Segoe UI", size=13, weight="bold")
        f_med = tkfont.Font(family="Segoe UI", size=10)

        # camera canvas
        self.cam  = tk.Canvas(self, width=CAM_W, height=CAM_H,
                              bg="black", highlightthickness=0)
        self.cam.pack()
        self._img = None

        # info panel
        panel = tk.Frame(self, bg="#0d0d1a", height=self.PANEL_H)
        panel.pack(fill="x"); panel.pack_propagate(False)

        left = tk.Frame(panel, bg="#0d0d1a"); left.pack(side="left", padx=12, pady=4)

        self.lbl_h  = tk.Label(left, text="Hands:   0",          font=f_med, bg="#0d0d1a", fg="#ff6060", anchor="w"); self.lbl_h.pack(anchor="w")
        self.lbl_r  = tk.Label(left, text="Raised:  ✗",          font=f_med, bg="#0d0d1a", fg="#ff6060", anchor="w"); self.lbl_r.pack(anchor="w")
        self.lbl_sw = tk.Label(left, text=f"Swings:  0/{SWINGS_NEEDED}", font=f_med, bg="#0d0d1a", fg="#50c8ff", anchor="w"); self.lbl_sw.pack(anchor="w")
        self.lbl_ax = tk.Label(left, text="X:0  Y:0  Z:0",       font=f_med, bg="#0d0d1a", fg="#888899", anchor="w"); self.lbl_ax.pack(anchor="w")

        right = tk.Frame(panel, bg="#0d0d1a"); right.pack(side="right", padx=12)
        self.lbl_status = tk.Label(right, text="Show hands to camera", font=f_big, bg="#0d0d1a", fg="#505080"); self.lbl_status.pack(anchor="e")
        self.lbl_hint   = tk.Label(right, text="↔ left/right  ↕ up/down  ⬡ closer/further", font=f_med, bg="#0d0d1a", fg="#334"); self.lbl_hint.pack(anchor="e")
        self.lbl_flash  = tk.Label(right, text="", font=f_big, bg="#0d0d1a", fg="#00ff99"); self.lbl_flash.pack(anchor="e")

        self._frame = None; self._state = {}
        self._lock  = threading.Lock(); self._fjob = None

        self.tracker = HandTracker(self._on_frame, self._on_state)
        self.tracker.start()
        self._poll()

    def _on_frame(self, bgr):
        with self._lock: self._frame = bgr.copy()

    def _on_state(self, s):
        with self._lock: self._state = dict(s)

    def _poll(self):
        with self._lock:
            frame = self._frame
            s     = dict(self._state)

        if frame is not None:
            self._img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.cam.create_image(0, 0, anchor="nw", image=self._img)

        if s.get("error"):
            self.lbl_status.config(text=s["error"], fg="#ff4444")
        else:
            n      = s.get("hands_detected", 0)
            raised = s.get("both_raised", False)
            total  = s.get("total", 0)
            cx, cy, cz = s.get("cx",0), s.get("cy",0), s.get("cz",0)

            self.lbl_h.config( text=f"Hands:  {n}",  fg="#00ff99" if n>=2 else "#ff6060")
            self.lbl_r.config( text="Raised: ✓" if raised else "Raised: ✗", fg="#00ff99" if raised else "#ff6060")
            self.lbl_sw.config(text=f"Swings: {total}/{SWINGS_NEEDED}", fg="#00ff99" if total>=SWINGS_NEEDED else "#50c8ff")
            self.lbl_ax.config(text=f"X:{cx}  Y:{cy}  Z:{cz}", fg="#aaaacc" if raised else "#444455")

            if raised:
                self.lbl_status.config(text="Swing any direction! 🙌", fg="#ffdd44")
                self.lbl_hint.config(fg="#556677")
            elif n > 0:
                self.lbl_status.config(text="Raise BOTH hands higher ↑", fg="#ff9944")
                self.lbl_hint.config(fg="#334")
            else:
                self.lbl_status.config(text="Show your hands to camera", fg="#505080")
                self.lbl_hint.config(fg="#334")

            if s.get("triggered"):
                self.lbl_flash.config(text="🎵  Playing 67.mp3!", fg="#00ff99")
                if self._fjob: self.after_cancel(self._fjob)
                self._fjob = self.after(1800, lambda: self.lbl_flash.config(text=""))

        self.after(30, self._poll)

    def _quit(self):
        self.tracker.stop()
        self.destroy()


if __name__ == "__main__":
    App().mainloop()