import csv
import json
import math
import os
import time
import tkinter as tk
from collections import deque
from datetime import datetime
from tkinter import ttk

import cv2
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageTk

# ----------------------------------------------------------------------------- settings you may change
BASE = os.path.dirname(os.path.abspath(__file__))          # the folder this script is in
MODEL_FILES = {                                            # name shown in the menu -> file name
    "Crop CNN 4 (no augmentation)": "crop_cnn4.h5",
    "Crop + augmentation CNN 4": "aug_cnn4.h5",
}
CAMERA_INDEX = 0          # 0 = built-in laptop camera; try 1 if you have a second camera
ROI_ASPECT = 3 / 4        # box width / height. The training photos were mostly portrait 3:4, so the box is too
SMOOTH_FRAMES = 5         # the displayed prediction averages the last 5 frames (the accuracy test does NOT smooth)
COUNTDOWN_SECONDS = 3     # time to get your hand into position before recording starts
MIN_SKIN_PERCENT = 15     # "No hand" is shown when less than this % of the box looks like skin (adjustable with a slider)


# ----------------------------------------------------------------------------- the model
# The .h5 file stores only the weight numbers, so the layer layout is rebuilt here.
# It must be identical to the one used in training (same names: branches / net / head).
WIDTHS = [32, 64, 128, 128, 256]


def conv_block(c_in, c_out, k):
    return nn.Sequential(nn.Conv2d(c_in, c_out, k, padding=k // 2), nn.BatchNorm2d(c_out), nn.ReLU(), nn.MaxPool2d(2))


class Branch(nn.Module):
    def __init__(self, k, blocks):
        super().__init__()
        ch = [3] + WIDTHS[:blocks]
        self.net = nn.Sequential(*[conv_block(ch[i], ch[i + 1], k) for i in range(blocks)],
                                 nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.out = ch[-1]

    def forward(self, x):
        return self.net(x)


class Net(nn.Module):
    def __init__(self, branches, n_classes):
        super().__init__()
        self.branches = nn.ModuleList(branches)
        self.head = nn.Sequential(nn.Dropout(0.4), nn.Linear(sum(b.out for b in branches), 64), nn.ReLU(),
                                  nn.Linear(64, n_classes))

    def forward(self, x):
        return self.head(torch.cat([b(x) for b in self.branches], dim=1))


def build_model(model_type, blocks, n_classes, input_size):
    if model_type == "linear":
        return nn.Sequential(nn.Flatten(), nn.Linear(3 * input_size * input_size, n_classes))
    if model_type == "cnn":
        return Net([Branch(3, blocks)], n_classes)
    if model_type == "twobranch":
        return Net([Branch(3, blocks), Branch(7, blocks)], n_classes)
    raise SystemExit(f"Unknown model type in the .h5 file: {model_type}")


def load_h5_model(path):
    """Read a .h5 file written by the training notebook. Returns (model, settings)."""
    with h5py.File(path, "r") as f:
        meta = json.loads(f.attrs["meta"])
        state = {k: torch.as_tensor(np.asarray(f["weights"][k][()])) for k in f["weights"]}
    model = build_model(meta["model"], meta["blocks"], len(meta["classes"]), meta["input_size"])
    model.load_state_dict(state)
    model.eval()                      # evaluation mode (no dropout)
    return model, meta


# ----------------------------------------------------------------------------- image preparation (same as training)
def hand_crop(rgb, margin=0.25):
    """
    Loose square crop around the biggest skin-coloured area (copy of the training code).
    Returns (crop, found, (x0, y0, side)).  found=False means nothing sensible was seen, so the centre was cut instead.
    """
    H, W = rgb.shape[:2]
    scale = 320 / max(H, W)                                   # find the hand on a small copy (fast)
    small = cv2.resize(rgb, (max(1, int(W * scale)), max(1, int(H * scale))), interpolation=cv2.INTER_AREA)
    sh, sw = small.shape[:2]

    ycrcb = cv2.cvtColor(small, cv2.COLOR_RGB2YCrCb)          # colour space where skin sits in a narrow range
    mask = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))  # 255 where the pixel looks like skin
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))                              # remove specks
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))  # fill small gaps
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    box = None
    if contours:
        areas = [cv2.contourArea(c) for c in contours]
        biggest = max(areas)
        if 0.02 <= biggest / (sh * sw) <= 0.85:               # skin region must be 2%..85% of the picture
            keep = [c for c, a in zip(contours, areas) if a >= 0.2 * biggest]
            x, y, w, h = cv2.boundingRect(np.concatenate(keep))
            box = (x / scale, y / scale, w / scale, h / scale)

    if box is None:
        side, cx, cy, found = float(min(H, W)), W / 2, H / 2, False
    else:
        x, y, w, h = box
        cx, cy = x + w / 2, y + h / 2
        side, found = min(max(w, h) * (1 + 2 * margin), float(min(H, W))), True
    x0 = int(round(min(max(cx - side / 2, 0), W - side)))
    y0 = int(round(min(max(cy - side / 2, 0), H - side)))
    side = int(round(side))
    return rgb[y0:y0 + side, x0:x0 + side], found, (x0, y0, side)


def identity_theta(n):
    t = torch.zeros(n, 2, 3)
    t[:, 0, 0] = 1
    t[:, 1, 1] = 1
    return t


def resample(x, theta, out):
    """The resize used in training (and for validation / test images): N x 3 x H x W -> N x 3 x out x out."""
    grid = F.affine_grid(theta, (x.size(0), x.size(1), out, out), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)


def make_work_copy(frame_bgr, box, margin, work_size):
    """
    Steps 1-3: cut the box out of the camera frame (BGR -> RGB), find the hand, crop around it, resize to the working size.
    Returns (work copy as uint8 RGB, hand found?, crop rectangle (x0, y0, side) in FRAME coordinates).
    """
    bx0, by0, bx1, by1 = box
    rgb = cv2.cvtColor(frame_bgr[by0:by1, bx0:bx1], cv2.COLOR_BGR2RGB)
    crop, found, (cx0, cy0, side) = hand_crop(rgb, margin)
    work = np.array(Image.fromarray(crop).resize((work_size, work_size), Image.LANCZOS))
    return work, found, (bx0 + cx0, by0 + cy0, side)


def to_model_input(work, meta):
    """
    Steps 4-5 for one model: resize to 128 x 128, then normalise with that model's own mean / std.
    Returns (tensor for the model, the 128 x 128 picture as a PIL image, for display).
    """
    x = torch.from_numpy(work).permute(2, 0, 1)[None].float() / 255
    x = resample(x, identity_theta(1), meta["input_size"])
    mean = torch.tensor(meta["mean"]).view(1, 3, 1, 1)
    std = torch.tensor(meta["std"]).view(1, 3, 1, 1)
    shown = Image.fromarray((x[0].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8))
    return (x - mean) / std, shown


def get_box(frame_shape, fraction):
    """The hand box: centred, 3:4 portrait, 'fraction' of the frame height tall. Returns x0, y0, x1, y1."""
    h, w = frame_shape[:2]
    bh = int(h * fraction)
    bw = int(bh * ROI_ASPECT)
    if bw > w:                                   # very narrow frame: shrink the box to fit
        bw = w
        bh = int(bw / ROI_ASPECT)
    x0, y0 = (w - bw) // 2, (h - bh) // 2
    return x0, y0, x0 + bw, y0 + bh


def skin_share(crop_bgr):
    """
    How much of the box looks like skin: the size of the biggest skin-coloured area, as a fraction (0..1) of the box.
    Used to decide whether a hand is in the box at all. It is a colour rule, not a hand detector:
    warm brown things (a wooden door, a table) can partly fool it, which is why the threshold has a slider.
    """
    small = cv2.resize(crop_bgr, (96, 128), interpolation=cv2.INTER_AREA)         # the box is 3:4
    ycrcb = cv2.cvtColor(small, cv2.COLOR_BGR2YCrCb)
    mask = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))                       # 255 where the pixel looks like skin
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))       # remove specks
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask)                        # separate skin areas
    return 0.0 if n <= 1 else float(stats[1:, cv2.CC_STAT_AREA].max()) / mask.size


# ----------------------------------------------------------------------------- accuracy bookkeeping
class Tally:
    """A confusion matrix: rows = the gesture you really showed, columns = what the model said."""

    def __init__(self, classes):
        self.classes = classes
        self.m = np.zeros((len(classes), len(classes)), dtype=int)

    def add(self, true_idx, pred_idx):
        self.m[true_idx, pred_idx] += 1

    def total(self):
        return int(self.m.sum())

    def accuracy(self):
        return self.m.trace() / self.total() if self.total() else None

    def recall(self, i):                          # of the frames where you showed class i, how many were right
        row = self.m[i].sum()
        return self.m[i, i] / row if row else None

    def reset(self):
        self.m[:] = 0


def pct(x):
    return "  n/a" if x is None else f"{x * 100:5.1f}%"


# ----------------------------------------------------------------------------- the window
class App:
    def __init__(self):
        # ---- load the models
        self.models, self.metas, self.classes = {}, {}, None
        for name, fname in MODEL_FILES.items():
            path = os.path.join(BASE, fname)
            if not os.path.exists(path):
                raise SystemExit(f"Model file not found: {path}\nPut the .h5 files next to this script.")
            model, meta = load_h5_model(path)
            if "margin" not in meta or "mean" not in meta:
                raise SystemExit(f"{fname} was not made by the crop + augmentation notebook (no crop settings inside).")
            if self.classes is None:
                self.classes, self.first = meta["classes"], meta
            elif (meta["classes"] != self.classes or meta["margin"] != self.first["margin"]
                  or meta["work_size"] != self.first["work_size"] or meta["input_size"] != self.first["input_size"]):
                raise SystemExit("The models were trained with different classes or crop settings; they cannot be compared.")
            self.models[name], self.metas[name] = model, meta
        self.margin, self.work_size = self.first["margin"], self.first["work_size"]
        self.tally = {n: Tally(self.classes) for n in self.models}
        self.history = {n: deque(maxlen=SMOOTH_FRAMES) for n in self.models}

        # ---- open the camera
        self.cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(CAMERA_INDEX)
        if not self.cap.isOpened():
            raise SystemExit("Could not open the camera. Close other apps that use it, or change CAMERA_INDEX.")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # ---- test state:  idle -> countdown -> collecting -> idle
        self.state, self.t_end, self.collected, self.target, self.true_idx = "idle", 0.0, 0, 0, 0
        self.skipped, self.skipped_at_start = 0, 0       # test frames with no hand in the box (not scored)

        self.build_window()
        self.refresh_results()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.update()

    # ------------------------------------------------------------------ window layout
    def build_window(self):
        self.root = tk.Tk()
        self.root.title("Rock Paper Scissors")

        self.model_name = tk.StringVar(value=list(self.models)[0])
        self.mirror = tk.BooleanVar(value=True)
        self.flip_input = tk.BooleanVar(value=False)
        self.box_frac = tk.DoubleVar(value=0.85)
        self.truth = tk.StringVar(value=self.classes[0])
        self.n_frames = tk.IntVar(value=100)
        self.status = tk.StringVar(value="Ready")
        self.gate_on = tk.BooleanVar(value=True)               # the "no hand" check
        self.min_skin = tk.DoubleVar(value=MIN_SKIN_PERCENT)   # threshold in percent
        self.skin_text = tk.StringVar(value="")
        self.crop_text = tk.StringVar(value="")

        left = tk.Frame(self.root)
        left.grid(row=0, column=0, padx=8, pady=8, sticky="n")   # sticky="n": video at the top of the window
        self.video = tk.Label(left)
        self.video.pack()

        right = tk.Frame(self.root)
        right.grid(row=0, column=1, padx=8, pady=8, sticky="n")

        # model choice and display options
        ttk.Label(right, text="Model").pack(anchor="w")
        box = ttk.Combobox(right, textvariable=self.model_name, values=list(self.models), state="readonly", width=30)
        box.pack(fill="x")
        box.bind("<<ComboboxSelected>>", self.on_model_change)
        ttk.Checkbutton(right, text="Mirror the preview (looks like a mirror)", variable=self.mirror).pack(anchor="w")
        ttk.Checkbutton(right, text="Also mirror the image given to the model", variable=self.flip_input).pack(anchor="w")
        ttk.Label(right, text="Box size").pack(anchor="w")
        ttk.Scale(right, from_=0.5, to=1.0, variable=self.box_frac).pack(fill="x")
        ttk.Checkbutton(right, text='Show "No hand" when the box has too little skin', variable=self.gate_on).pack(anchor="w")
        ttk.Label(right, textvariable=self.skin_text).pack(anchor="w")
        ttk.Scale(right, from_=2, to=60, variable=self.min_skin).pack(fill="x")      # skin % needed to count as a hand
        ttk.Label(right, textvariable=self.crop_text).pack(anchor="w")

        # prediction
        self.pred_label = tk.Label(right, text="-", font=("Arial", 26, "bold"))
        self.pred_label.pack(pady=(10, 4))
        self.bar_text, self.bars = [], []
        for c in self.classes:
            row = tk.Frame(right)
            row.pack(fill="x")
            t = tk.Label(row, text=c, width=9, anchor="w")
            t.pack(side="left")
            b = ttk.Progressbar(row, length=150, maximum=100)
            b.pack(side="left", padx=4)
            v = tk.Label(row, text="0%", width=5)
            v.pack(side="left")
            self.bars.append(b)
            self.bar_text.append(v)

        # what the model actually sees
        ttk.Label(right, text="What the model sees (cropped, 128 x 128)").pack(anchor="w", pady=(10, 0))
        self.seen = tk.Label(right)
        self.seen.pack()

        # accuracy test
        ttk.Separator(right).pack(fill="x", pady=8)
        ttk.Label(right, text="ACCURACY TEST: I am showing", font=("Arial", 10, "bold")).pack(anchor="w")
        for c in self.classes:
            ttk.Radiobutton(right, text=c, value=c, variable=self.truth).pack(anchor="w")
        row = tk.Frame(right)
        row.pack(fill="x", pady=4)
        ttk.Label(row, text="Frames to record").pack(side="left")
        ttk.Spinbox(row, from_=20, to=1000, increment=20, textvariable=self.n_frames, width=6).pack(side="left", padx=4)
        ttk.Button(right, text="Start test", command=self.start_test).pack(fill="x")
        ttk.Button(right, text="Reset results", command=self.reset_results).pack(fill="x", pady=2)
        ttk.Button(right, text="Save results (CSV)", command=self.save_csv).pack(fill="x")
        ttk.Label(right, textvariable=self.status, foreground="blue").pack(anchor="w", pady=4)

        self.results = tk.Label(right, text="", font=("Courier", 9), justify="left", anchor="w")
        self.results.pack(anchor="w")

    # ------------------------------------------------------------------ controls
    def on_model_change(self, event=None):
        for h in self.history.values():     # forget the old averaging window
            h.clear()
        self.refresh_results()              # the confusion matrix shown belongs to the selected model

    def start_test(self):
        if self.state != "idle":
            return
        self.true_idx = self.classes.index(self.truth.get())
        self.target = max(1, int(self.n_frames.get()))
        self.skipped_at_start = self.skipped
        self.t_end = time.time() + COUNTDOWN_SECONDS
        self.state = "countdown"

    def reset_results(self):
        for t in self.tally.values():
            t.reset()
        self.skipped = self.skipped_at_start = 0
        self.status.set("Results cleared")
        self.refresh_results()

    def save_csv(self):
        path = os.path.join(BASE, f"webcam_test_results_{datetime.now():%Y%m%d_%H%M%S}.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["model", "true_gesture", "predicted_gesture", "frames"])
            for name, t in self.tally.items():
                for i, true_c in enumerate(self.classes):
                    for j, pred_c in enumerate(self.classes):
                        w.writerow([name, true_c, pred_c, int(t.m[i, j])])
        self.status.set(f"Saved {os.path.basename(path)}")

    def refresh_results(self):
        lines = []
        for name, t in self.tally.items():
            acc = t.accuracy()
            head = "no frames yet" if acc is None else f"{acc * 100:.1f}%  ({t.m.trace()} of {t.total()} frames)"
            lines.append(f"{name}: {head}")
            lines.append("  " + "  ".join(f"{c} {pct(t.recall(i))}" for i, c in enumerate(self.classes)))
        lines.append(f"Test frames skipped (no hand in box): {self.skipped}")
        sel = self.model_name.get()
        lines += ["", f"Confusion matrix: {sel}", "rows = shown, columns = predicted"]
        lines.append(" " * 10 + "".join(f"{c[:8]:>9}" for c in self.classes))
        for i, c in enumerate(self.classes):
            lines.append(f"{c[:9]:<10}" + "".join(f"{int(v):>9}" for v in self.tally[sel].m[i]))
        self.results.config(text="\n".join(lines))

    # ------------------------------------------------------------------ one camera frame
    def update(self):
        ok, frame = self.cap.read()
        if not ok:
            self.status.set("No camera image; retrying...")
            self.root.after(300, self.update)
            return

        W = frame.shape[1]
        box = get_box(frame.shape, self.box_frac.get())
        frame_in = cv2.flip(frame, 1) if self.flip_input.get() else frame     # what the models are given

        # steps 1-3 once per frame (shared by all models): box -> hand crop -> 256 x 256 working copy
        work, found, (cx0, cy0, side) = make_work_copy(frame_in, box, self.margin, self.work_size)

        # is there a hand in the box at all?  (the models always answer Rock/Paper/Scissors, even for an empty wall)
        share = skin_share(frame_in[box[1]:box[3], box[0]:box[2]])
        need = self.min_skin.get() / 100
        has_hand = (not self.gate_on.get()) or share >= need
        self.skin_text.set(f"Skin in box: {share * 100:.0f}%   (needs {need * 100:.0f}%)")
        self.crop_text.set("Crop: hand found" if found else "Crop: no hand found, centre of the box used")

        # steps 4-5 and the prediction, for every model on this same frame
        probs, shown = {}, None
        with torch.no_grad():
            for name, model in self.models.items():
                x, img = to_model_input(work, self.metas[name])
                shown = img if shown is None else shown
                probs[name] = torch.softmax(model(x), dim=1)[0].numpy()
                if has_hand:
                    self.history[name].append(probs[name])

        # the accuracy test uses the RAW per-frame prediction of every model (no smoothing)
        if self.state == "countdown":
            left = math.ceil(self.t_end - time.time())
            if left <= 0:
                self.state, self.collected = "collecting", 0
            else:
                self.status.set(f"Get ready to show {self.classes[self.true_idx]}: {left}")
        if self.state == "collecting":
            if has_hand:
                for name in self.models:
                    self.tally[name].add(self.true_idx, int(np.argmax(probs[name])))
            else:
                self.skipped += 1                    # no hand in the box: this frame is not scored
            self.collected += 1
            self.status.set(f"Recording {self.classes[self.true_idx]}: {self.collected}/{self.target}")
            if self.collected >= self.target:
                self.state = "idle"
                skipped_now = self.skipped - self.skipped_at_start
                self.status.set(f"Done: {self.target} frames of {self.classes[self.true_idx]}"
                                + (f" ({skipped_now} skipped: no hand in the box)" if skipped_now else ""))
            self.refresh_results()

        # displayed prediction = average of the last few frames of the selected model
        sel = self.model_name.get()
        if has_hand:
            avg = np.mean(self.history[sel], axis=0)
            k = int(np.argmax(avg))
            self.pred_label.config(text=f"{self.classes[k]}  {avg[k] * 100:.0f}%", fg="black")
            for i in range(len(self.classes)):
                self.bars[i]["value"] = avg[i] * 100
                self.bar_text[i].config(text=f"{avg[i] * 100:.0f}%")
        else:
            self.pred_label.config(text="No hand", fg="gray")
            for i in range(len(self.classes)):
                self.bars[i]["value"] = 0
                self.bar_text[i].config(text="-")

        # ---- draw the preview: green box = where to hold the hand; blue square = the crop the models receive
        disp = cv2.flip(frame, 1) if self.mirror.get() else frame.copy()
        color = (0, 255, 255) if self.state != "idle" else (0, 200, 0)
        cv2.rectangle(disp, (box[0], box[1]), (box[2], box[3]), color, 3)       # the box is centred, so it is the same mirrored
        a, b = cx0, cx0 + side                                                  # crop columns, in frame_in coordinates
        if self.flip_input.get():
            a, b = W - b, W - a                                                 # back to the original camera frame
        if self.mirror.get():
            a, b = W - b, W - a                                                 # and into the mirrored preview
        cv2.rectangle(disp, (a, cy0), (b, cy0 + side), (255, 120, 0) if found else (0, 0, 255), 2)
        if self.state == "countdown":
            cv2.putText(disp, str(max(1, math.ceil(self.t_end - time.time()))), (disp.shape[1] // 2 - 25, disp.shape[0] // 2 + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 6)
        photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)))
        self.video.configure(image=photo)
        self.video.image = photo                                                # keep a reference or it disappears
        seen = ImageTk.PhotoImage(shown.resize((192, 192), Image.NEAREST))
        self.seen.configure(image=seen)
        self.seen.image = seen

        self.root.after(10, self.update)

    def close(self):
        self.cap.release()
        self.root.destroy()


if __name__ == "__main__":
    App().root.mainloop()