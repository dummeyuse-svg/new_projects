
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
import os
import json
from datetime import datetime
from collections import deque
from PIL import Image, ImageTk

# ── Picamera2 optional import ──────────────────────────────────────
try:
    from picamera2 import Picamera2
    PICAMERA_AVAILABLE = True
except ImportError:
    PICAMERA_AVAILABLE = False

# ──────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────
CONFIG = {
    # Camera
    "CAPTURE_RESOLUTION": (2304, 1296),   # Picamera2 only
    "LENS_POSITION":       2.0,
    "EXPOSURE_TIME":       5000,
    "WEBCAM_INDEX":        0,             # laptop webcam index

    # Image processing
    "TARGET_WIDTH":        1280,          # resize all frames to this width

    # ROI (in normalised image coordinates after TARGET_WIDTH resize)
    "ROI_Y_START":         200,
    "ROI_Y_END":           500,
    "ROI_X_START":         50,
    "ROI_X_END":           1230,

    # ── Detection thresholds ──────────────────────────────────────
    # These are the values you should tune for your physical setup.
    # Start conservative (high thresholds) and lower them until you
    # catch real uplift without false fails.

    # Per-column flag threshold (pixels). At PX_PER_MM=10 → 0.3 mm.
    # Must be LOWER than MAX_MM_THRESHOLD × PX_PER_MM.
    "UPLIFT_THRESHOLD_PX":  3,

    # Minimum consecutive flagged columns to form a region (noise filter).
    "MIN_FAIL_COLUMNS":     25,

    # Overall fail: peak uplift in ANY region must exceed this (mm).
    "MAX_MM_THRESHOLD":     0.8,

    # Pixels per millimetre — CALIBRATE THIS to your actual setup.
    # Measure a known object (e.g. 10 mm reference block) and count pixels.
    "PX_PER_MM":            10.0,

    # A region must have mean uplift ≥ this to avoid wide-shallow noise.
    # Raised from 0.4 to 0.5 to reduce false fails.
    "MIN_REGION_MEAN_MM":   0.5,

    # ── Edge detection ────────────────────────────────────────────
    "BLUR_KERNEL":      9,
    "CANNY_LOW":        15,
    "CANNY_HIGH":       60,
    "EDGE_PERCENTILE":  20,   # percentile of edge pixels = board surface

    # ── Temporal smoothing ────────────────────────────────────────
    # Both baseline AND live profiles are median-smoothed over N frames.
    # This is the single most important fix for noise parity.
    "TEMPORAL_FRAMES":  20,   # live inspection buffer
    "BASELINE_FRAMES":  25,   # frames used when calibrating
    "LIVE_INSPECT_EVERY": 1,  # analyse every N-th live frame

    # ── Detrending ────────────────────────────────────────────────
    # Order 1 = linear tilt only. Order 2 adds gentle bow.
    # Use 1 unless you have strong lens distortion.
    "DETREND_ORDER":    1,
    # Fraction of flattest columns to use for detrend fit.
    # Higher = more columns included; lower = stricter inlier selection.
    "DETREND_INLIER_FRACTION": 0.60,

    # Files
    "BASELINE_FILE":   "baseline.json",
    "ROI_CONFIG_FILE": "roi_config.json",
    "LOG_DIR":         "inspection_logs",
}

# ──────────────────────────────────────────────
# CAMERA BACKEND DETECTION
# ──────────────────────────────────────────────
def detect_camera_backend():
    """Return 'picamera2', 'webcam', or None."""
    if PICAMERA_AVAILABLE:
        return 'picamera2'
    # Try to open webcam
    cap = cv2.VideoCapture(CONFIG["WEBCAM_INDEX"])
    if cap.isOpened():
        cap.release()
        return 'webcam'
    return None

CAMERA_BACKEND = detect_camera_backend()
CAMERA_AVAILABLE = CAMERA_BACKEND is not None

# ──────────────────────────────────────────────
# ROI CONFIG
# ──────────────────────────────────────────────
def load_roi_config():
    if os.path.exists(CONFIG["ROI_CONFIG_FILE"]):
        with open(CONFIG["ROI_CONFIG_FILE"]) as f:
            roi = json.load(f)
        for key in ("ROI_X_START", "ROI_X_END", "ROI_Y_START", "ROI_Y_END"):
            if key in roi:
                CONFIG[key] = roi[key]

def save_roi_config():
    roi = {k: CONFIG[k] for k in ("ROI_X_START", "ROI_X_END", "ROI_Y_START", "ROI_Y_END")}
    with open(CONFIG["ROI_CONFIG_FILE"], "w") as f:
        json.dump(roi, f, indent=2)

# ──────────────────────────────────────────────
# IMAGE NORMALISATION
# ──────────────────────────────────────────────
def normalize_image(bgr_img):
    h, w  = bgr_img.shape[:2]
    scale = CONFIG["TARGET_WIDTH"] / w
    new_h = int(h * scale)
    return cv2.resize(bgr_img, (CONFIG["TARGET_WIDTH"], new_h),
                      interpolation=cv2.INTER_AREA)

# ──────────────────────────────────────────────
# EDGE PROFILE EXTRACTION
# ──────────────────────────────────────────────
def get_edge_profile(gray_img):
    """
    Extract the top-edge profile of the PCB/jig within the ROI.

    Returns a 1-D float array of length (ROI_X_END - ROI_X_START),
    where each value is the ROW position of the board's top edge
    in full-image pixel coordinates.

    SIGN NOTE: smaller row = higher in the image = board is lifted.
    """
    roi = gray_img[
        CONFIG["ROI_Y_START"]:CONFIG["ROI_Y_END"],
        CONFIG["ROI_X_START"]:CONFIG["ROI_X_END"]
    ]
    roi_h, roi_w = roi.shape

    if roi_h < 5 or roi_w < 5:
        raise ValueError("ROI is too small. Please redraw it.")

    # CLAHE normalisation
    clahe  = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    roi_eq = clahe.apply(roi)

    # Gaussian blur
    k = CONFIG["BLUR_KERNEL"]
    if k % 2 == 0:
        k += 1
    blurred = cv2.GaussianBlur(roi_eq, (k, k), 0)

    # Canny edge detection
    edges = cv2.Canny(blurred, CONFIG["CANNY_LOW"], CONFIG["CANNY_HIGH"])

    # Close small gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges  = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # Per-column edge extraction: only search top 60% of ROI
    search_limit    = int(roi_h * 0.60)
    profile         = np.full(roi_w, fill_value=float(roi_h * 0.5), dtype=np.float32)
    EDGE_PERCENTILE = CONFIG["EDGE_PERCENTILE"]

    for col in range(roi_w):
        col_edges = np.where(edges[:search_limit, col] > 0)[0]
        if len(col_edges) >= 3:
            profile[col] = float(np.percentile(col_edges, EDGE_PERCENTILE))
        elif len(col_edges) >= 1:
            profile[col] = float(col_edges[0])

    # Convert to absolute image row coordinates
    profile += CONFIG["ROI_Y_START"]

    # Spike rejection
    profile = _spike_filter(profile, window=31, sigma_thresh=2.5)

    # Savitzky-Golay smoothing
    profile = _savgol_smooth(profile, window=25, poly=3)

    return profile


def _spike_filter(profile, window=31, sigma_thresh=2.5):
    out  = profile.copy()
    half = window // 2
    n    = len(profile)
    for i in range(n):
        lo  = max(0, i - half)
        hi  = min(n, i + half + 1)
        seg = profile[lo:hi]
        med = np.median(seg)
        std = np.std(seg)
        if std > 0 and abs(profile[i] - med) > sigma_thresh * std:
            out[i] = med
    return out


def _savgol_smooth(profile, window=25, poly=3):
    n = len(profile)
    if n < window:
        return profile
    if window % 2 == 0:
        window += 1
    try:
        from scipy.signal import savgol_filter
        return savgol_filter(profile, window_length=window, polyorder=poly).astype(np.float32)
    except ImportError:
        half   = window // 2
        kernel = np.ones(window, dtype=np.float32) / window
        padded = np.pad(profile, half, mode='edge')
        return np.convolve(padded, kernel, mode='valid').astype(np.float32)


# ──────────────────────────────────────────────
# ROBUST DETRENDING
# ──────────────────────────────────────────────
def robust_detrend(diff, order=1, inlier_fraction=0.60):
    """
    Remove camera tilt / lens bow WITHOUT removing real uplift signal.

    Two-pass iterative robust fit:
      Pass 1 — fit on ALL columns, compute residuals.
      Pass 2 — fit on only the FLATTEST `inlier_fraction` columns.
               Uplifted columns have large residuals from Pass 1 and
               are excluded, so they cannot corrupt the trend estimate.

    No median subtraction — that would zero out any uniform uplift.
    """
    n = len(diff)
    if n < order + 2:
        return diff.astype(np.float32)

    x = np.linspace(0, 1, n)

    # Pass 1: preliminary fit on all columns
    c1       = np.polyfit(x, diff, order)
    resid1   = diff - np.polyval(c1, x)

    # Pass 2: refit on flattest columns only
    thresh   = np.percentile(np.abs(resid1), inlier_fraction * 100)
    inliers  = np.abs(resid1) <= thresh

    if inliers.sum() < order + 2:
        # Too few inliers — just use pass-1 trend
        return (diff - np.polyval(c1, x)).astype(np.float32)

    c2    = np.polyfit(x[inliers], diff[inliers], order)
    trend = np.polyval(c2, x)

    return (diff - trend).astype(np.float32)


# ──────────────────────────────────────────────
# UPLIFT ANALYSIS
# ──────────────────────────────────────────────
def analyze_uplift(current_profile, baseline_profile):
    """
    Compare current edge profile against the baseline.

    Sign convention (row coordinates, smaller = higher):
        diff = baseline_row - current_row
        diff > 0  →  current edge is ABOVE baseline  →  board is LIFTED
        diff < 0  →  current edge is BELOW baseline  →  board is SUNKEN

    Pass/Fail gate (ALL three required for a fail region):
        1. Region width  ≥ MIN_FAIL_COLUMNS
        2. Region mean   ≥ MIN_REGION_MEAN_MM
        3. Region peak   ≥ MAX_MM_THRESHOLD

    Global fail also triggers if the raw peak exceeds MAX_MM_THRESHOLD
    even outside any qualifying region.
    """
    if len(current_profile) != len(baseline_profile):
        baseline_profile = np.interp(
            np.linspace(0, 1, len(current_profile)),
            np.linspace(0, 1, len(baseline_profile)),
            baseline_profile,
        )

    # Raw difference: positive means current edge is higher (lifted)
    raw_diff = baseline_profile.astype(np.float32) - current_profile.astype(np.float32)

    # Remove camera/lens geometry, preserve uplift signal
    detrended = robust_detrend(
        raw_diff,
        order            = CONFIG["DETREND_ORDER"],
        inlier_fraction  = CONFIG["DETREND_INLIER_FRACTION"],
    )

    abs_det  = np.abs(detrended)
    det_mm   = detrended   / CONFIG["PX_PER_MM"]
    abs_mm   = abs_det     / CONFIG["PX_PER_MM"]

    # Per-column flagging
    flagged    = abs_det > CONFIG["UPLIFT_THRESHOLD_PX"]
    max_abs_mm = float(np.max(abs_mm))

    # Region detection
    fail_regions  = []
    in_region     = False
    region_start  = 0
    total_cols    = len(flagged)

    for i in range(total_cols + 1):
        is_flag = (i < total_cols) and flagged[i]

        if is_flag and not in_region:
            in_region    = True
            region_start = i

        elif not is_flag and in_region:
            in_region    = False
            length       = i - region_start
            if length >= CONFIG["MIN_FAIL_COLUMNS"]:
                seg          = detrended[region_start:i]
                seg_mm       = det_mm[region_start:i]
                mean_abs_mm  = float(np.mean(np.abs(seg_mm)))
                peak_mm      = float(np.max(np.abs(seg_mm)))

                # ALL THREE gates must pass
                if (mean_abs_mm >= CONFIG["MIN_REGION_MEAN_MM"] and
                        peak_mm >= CONFIG["MAX_MM_THRESHOLD"]):
                    fail_regions.append({
                        "col_start":      region_start,
                        "col_end":        i,
                        "max_uplift_px":  float(np.max(np.abs(seg))),
                        "max_uplift_mm":  peak_mm,
                        "mean_uplift_mm": mean_abs_mm,
                        "direction":      "lifted" if float(np.mean(seg)) > 0 else "sunken",
                        "x_start_mm":     (region_start / total_cols) * 200.0,
                        "x_end_mm":       (i            / total_cols) * 200.0,
                    })

    passed = (len(fail_regions) == 0) and (max_abs_mm <= CONFIG["MAX_MM_THRESHOLD"])
    return passed, fail_regions, detrended, det_mm, max_abs_mm


# ──────────────────────────────────────────────
# TEMPORAL PROFILE SMOOTHER
# ──────────────────────────────────────────────
class TemporalProfileSmoother:
    """Per-column median of the last N edge profiles."""

    def __init__(self, n_frames=20):
        self._n      = n_frames
        self._buffer = deque(maxlen=n_frames)

    def push(self, profile):
        self._buffer.append(profile.copy())

    def get_smoothed(self):
        if not self._buffer:
            return None
        stack = np.stack(list(self._buffer), axis=0)
        return np.median(stack, axis=0).astype(np.float32)

    def reset(self):
        self._buffer.clear()

    @property
    def ready(self):
        return len(self._buffer) >= max(3, int(self._n * 0.5))

    @property
    def fill_fraction(self):
        return len(self._buffer) / self._n


# ──────────────────────────────────────────────
# BASELINE COLLECTOR
# ──────────────────────────────────────────────
class BaselineCollector:
    """Collects N frames and returns their per-column median as the baseline."""

    def __init__(self, n_frames=25):
        self._n      = n_frames
        self._buffer = []

    def add_frame(self, profile):
        self._buffer.append(profile.copy())

    @property
    def complete(self):
        return len(self._buffer) >= self._n

    @property
    def progress(self):
        return len(self._buffer) / self._n

    def get_baseline(self):
        if not self._buffer:
            return None
        stack = np.stack(self._buffer, axis=0)
        return np.median(stack, axis=0).astype(np.float32)

    def reset(self):
        self._buffer = []


# ──────────────────────────────────────────────
# ANNOTATED IMAGE
# ──────────────────────────────────────────────
def build_annotated_image(color_bgr, current_profile, baseline_profile,
                           diff, fail_regions, passed):
    ann    = color_bgr.copy()
    roi_x0 = CONFIG["ROI_X_START"]
    abs_diff = np.abs(diff)

    for col_idx, by in enumerate(baseline_profile):
        x, y = col_idx + roi_x0, int(by)
        if 0 <= y < ann.shape[0] and 0 <= x < ann.shape[1]:
            ann[y, x] = (0, 255, 0)   # green = baseline

    for col_idx, cy in enumerate(current_profile):
        x, y  = col_idx + roi_x0, int(cy)
        color = (0, 0, 255) if abs_diff[col_idx] > CONFIG["UPLIFT_THRESHOLD_PX"] else (0, 255, 255)
        if 0 <= y < ann.shape[0] and 0 <= x < ann.shape[1]:
            ann[y, x] = color   # red=flagged, yellow=ok

    cv2.rectangle(ann,
                  (CONFIG["ROI_X_START"], CONFIG["ROI_Y_START"]),
                  (CONFIG["ROI_X_END"],   CONFIG["ROI_Y_END"]),
                  (0, 165, 255), 2)

    label = "PASS" if passed else f"FAIL — {len(fail_regions)} region(s)"
    color = (0, 200, 0) if passed else (0, 0, 255)
    cv2.putText(ann, label, (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 4)

    y_off = 150
    for r in fail_regions:
        d    = "lifted" if r["direction"] == "lifted" else "sunken"
        text = (f"  {d}  {r['max_uplift_mm']:.2f}mm "
                f"@ {r['x_start_mm']:.0f}–{r['x_end_mm']:.0f}mm")
        cv2.putText(ann, text, (50, y_off),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        y_off += 45
    return ann


# ──────────────────────────────────────────────
# BASELINE LOAD / SAVE
# ──────────────────────────────────────────────
def load_baseline():
    if not os.path.exists(CONFIG["BASELINE_FILE"]):
        raise FileNotFoundError("No baseline.json — run Calibrate first.")
    with open(CONFIG["BASELINE_FILE"]) as f:
        data = json.load(f)
    return np.array(data["baseline_per_col"], dtype=np.float32)

def save_baseline(profile, rgb_img):
    data = {
        "baseline_median_y": float(np.median(profile)),
        "baseline_per_col":  profile.tolist(),
        "timestamp":         datetime.now().isoformat(),
        "target_width":      CONFIG["TARGET_WIDTH"],
        "roi":               {k: CONFIG[k] for k in
                              ("ROI_X_START","ROI_X_END","ROI_Y_START","ROI_Y_END")},
        "n_frames_averaged": CONFIG["BASELINE_FRAMES"],
    }
    with open(CONFIG["BASELINE_FILE"], "w") as f:
        json.dump(data, f, indent=2)
    os.makedirs(CONFIG["LOG_DIR"], exist_ok=True)
    cv2.imwrite(
        os.path.join(CONFIG["LOG_DIR"], "calibration_image.jpg"),
        cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR),
    )


# ──────────────────────────────────────────────
# LOG ENTRY
# ──────────────────────────────────────────────
def write_log(timestamp, source, passed, max_abs_mm, fail_regions, img_path):
    os.makedirs(CONFIG["LOG_DIR"], exist_ok=True)
    entry = {
        "timestamp":    timestamp,
        "source":       source,
        "result":       "PASS" if passed else "FAIL",
        "max_mm":       round(max_abs_mm, 4),
        "fail_regions": fail_regions,
        "image":        img_path,
    }
    with open(os.path.join(CONFIG["LOG_DIR"], "inspection_log.jsonl"), "a") as f:
        f.write(json.dumps(entry) + "\n")


# ══════════════════════════════════════════════════════════════════
# GUI CONSTANTS
# ══════════════════════════════════════════════════════════════════
DARK_BG    = "#0d1117"
PANEL_BG   = "#161b22"
BORDER     = "#30363d"
TEXT_FG    = "#e6edf3"
MUTED      = "#8b949e"
ACCENT     = "#58a6ff"
GREEN      = "#3fb950"
RED        = "#f85149"
ORANGE     = "#d29922"
BTN_BG     = "#21262d"
BTN_HOV    = "#30363d"
FONT_MONO  = ("Courier New", 11)
FONT_LABEL = ("Segoe UI", 10)
FONT_TITLE = ("Segoe UI", 11, "bold")


def pil_from_bgr(bgr, max_w, max_h):
    h, w  = bgr.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    nw, nh = int(w * scale), int(h * scale)
    small  = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    rgb    = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    return ImageTk.PhotoImage(Image.fromarray(rgb))


# ──────────────────────────────────────────────
# ROI EDIT DIALOG
# ──────────────────────────────────────────────
class ROIEditDialog(tk.Toplevel):
    def __init__(self, parent, on_apply):
        super().__init__(parent)
        self.title("Edit ROI Coordinates")
        self.configure(bg=PANEL_BG)
        self.resizable(False, False)
        self.grab_set()
        self._on_apply = on_apply
        pad = dict(padx=12, pady=6)

        tk.Label(self, text="ROI COORDINATE EDITOR",
                 font=("Segoe UI", 11, "bold"), bg=PANEL_BG, fg=ACCENT
                 ).grid(row=0, column=0, columnspan=2, padx=12, pady=(14, 4))

        desc = (
            "Set the rectangular region of interest.\n"
            f"Coordinates are in pixels after resize to {CONFIG['TARGET_WIDTH']}px wide."
        )
        tk.Label(self, text=desc, font=("Segoe UI", 9), bg=PANEL_BG,
                 fg=MUTED, justify="left"
                 ).grid(row=1, column=0, columnspan=2, padx=12, pady=(0, 10))

        fields = [
            ("X Start  (left edge):",   "ROI_X_START"),
            ("X End    (right edge):",  "ROI_X_END"),
            ("Y Start  (top edge):",    "ROI_Y_START"),
            ("Y End    (bottom edge):", "ROI_Y_END"),
        ]
        self._vars = {}
        for row_idx, (label, key) in enumerate(fields, start=2):
            tk.Label(self, text=label, font=FONT_LABEL, bg=PANEL_BG,
                     fg=TEXT_FG, anchor="w"
                     ).grid(row=row_idx, column=0, sticky="w", **pad)
            var = tk.StringVar(value=str(CONFIG[key]))
            self._vars[key] = var
            tk.Entry(self, textvariable=var, width=10, bg=BTN_BG, fg=TEXT_FG,
                     insertbackground=ACCENT, relief="flat", font=FONT_MONO
                     ).grid(row=row_idx, column=1, sticky="ew", **pad)

        tk.Frame(self, bg=BORDER, height=1).grid(
            row=6, column=0, columnspan=2, sticky="ew", padx=10, pady=6)

        btn_frame = tk.Frame(self, bg=PANEL_BG)
        btn_frame.grid(row=7, column=0, columnspan=2, pady=(0, 12))

        tk.Button(btn_frame, text="Apply", command=self._apply,
                  bg=ACCENT, fg=DARK_BG, font=("Segoe UI", 10, "bold"),
                  relief="flat", padx=18, pady=6, cursor="hand2"
                  ).pack(side="left", padx=6)
        tk.Button(btn_frame, text="Cancel", command=self.destroy,
                  bg=BTN_BG, fg=TEXT_FG, font=("Segoe UI", 10),
                  relief="flat", padx=18, pady=6, cursor="hand2"
                  ).pack(side="left", padx=6)

        self.columnconfigure(1, weight=1)
        self.transient(parent)

    def _apply(self):
        try:
            vals = {k: int(v.get()) for k, v in self._vars.items()}
        except ValueError:
            messagebox.showerror("Invalid Input",
                                 "All coordinates must be integers.", parent=self)
            return
        if vals["ROI_X_START"] >= vals["ROI_X_END"]:
            messagebox.showerror("Invalid ROI", "X Start must be less than X End.", parent=self)
            return
        if vals["ROI_Y_START"] >= vals["ROI_Y_END"]:
            messagebox.showerror("Invalid ROI", "Y Start must be less than Y End.", parent=self)
            return
        for k, v in vals.items():
            CONFIG[k] = v
        save_roi_config()
        self._on_apply()
        self.destroy()


# ══════════════════════════════════════════════════════════════════
# WEBCAM CAPTURE THREAD
# ══════════════════════════════════════════════════════════════════
class WebcamCapture:
    """
    Wraps cv2.VideoCapture and provides the same interface used by the
    Picamera2 path: capture_array() returns an RGB numpy array.
    """
    def __init__(self, index=0):
        self._cap = cv2.VideoCapture(index)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open webcam index {index}")
        # Set high resolution if supported
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1920)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    def capture_array(self):
        ret, frame = self._cap.read()
        if not ret:
            raise RuntimeError("Webcam read failed")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)   # return RGB

    def stop(self):
        self._cap.release()


# ══════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ══════════════════════════════════════════════════════════════════
class PCBApp(tk.Tk):
    PREVIEW_W = 760
    PREVIEW_H = 430
    RESULT_W  = 760
    RESULT_H  = 430

    def __init__(self):
        super().__init__()
        self.title("PCB Warpage Detection System")
        self.configure(bg=DARK_BG)
        self.resizable(True, True)

        # Camera state
        self.cam              = None
        self.baseline_profile = None
        self.last_rgb         = None
        self.last_gray        = None
        self._preview_running = False
        self._preview_thread  = None
        self._live_frame_bgr  = None
        self._live_lock       = threading.Lock()

        # Live-inspect state
        self._live_inspect_active = False
        self._live_result         = None
        self._live_result_detail  = ""
        self._live_frame_count    = 0
        self._temporal_smoother   = TemporalProfileSmoother(CONFIG["TEMPORAL_FRAMES"])

        # Calibration state
        self._calibrating         = False
        self._baseline_collector  = BaselineCollector(CONFIG["BASELINE_FRAMES"])
        self._calib_rgb_sample    = None

        # ROI drawing state
        self._roi_drawing   = False
        self._roi_start     = None
        self._roi_rect_id   = None
        self._roi_mode      = False
        self._result_scale  = 1.0
        self._result_offset = (0, 0)

        load_roi_config()
        self._build_ui()
        self._try_load_baseline()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        if CAMERA_AVAILABLE:
            self._start_camera()
        else:
            self._log("⚠  No camera found — file upload mode only.")

    # ── UI ────────────────────────────────────────────────────────
    def _build_ui(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        # Header
        hdr = tk.Frame(self, bg=PANEL_BG)
        hdr.grid(row=0, column=0, sticky="ew")
        hdr.columnconfigure(1, weight=1)

        cam_label = f"[{CAMERA_BACKEND or 'no camera'}]"
        tk.Label(hdr, text=f"⬡  PCB WARPAGE INSPECTOR  {cam_label}",
                 font=("Segoe UI", 13, "bold"),
                 bg=PANEL_BG, fg=ACCENT, padx=16, pady=10
                 ).grid(row=0, column=0, sticky="w")

        self._status_var = tk.StringVar(value="Ready")
        tk.Label(hdr, textvariable=self._status_var, font=FONT_LABEL,
                 bg=PANEL_BG, fg=MUTED, padx=16
                 ).grid(row=0, column=1, sticky="e")

        tk.Frame(self, bg=BORDER, height=1).grid(row=0, column=0, sticky="ews")

        # Body
        body = tk.Frame(self, bg=DARK_BG)
        body.grid(row=1, column=0, sticky="nsew", padx=12, pady=10)
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=0)
        body.rowconfigure(0, weight=1)

        panels = tk.Frame(body, bg=DARK_BG)
        panels.grid(row=0, column=0, sticky="nsew")
        panels.columnconfigure(0, weight=1)
        panels.columnconfigure(1, weight=1)
        panels.rowconfigure(1, weight=1)

        tk.Label(panels, text="LIVE PREVIEW", font=FONT_TITLE,
                 bg=DARK_BG, fg=MUTED).grid(row=0, column=0, pady=(0,4), sticky="w", padx=4)
        tk.Label(panels, text="CAPTURE / RESULT", font=FONT_TITLE,
                 bg=DARK_BG, fg=MUTED).grid(row=0, column=1, pady=(0,4), sticky="w", padx=4)

        self._live_canvas = tk.Canvas(
            panels, width=self.PREVIEW_W, height=self.PREVIEW_H,
            bg="#0a0e14", highlightthickness=1, highlightbackground=BORDER)
        self._live_canvas.grid(row=1, column=0, padx=(0,6), sticky="nsew")
        self._live_canvas.create_text(
            self.PREVIEW_W//2, self.PREVIEW_H//2,
            text="Waiting for camera…", fill=MUTED, font=FONT_MONO, tags="placeholder")

        self._result_canvas = tk.Canvas(
            panels, width=self.RESULT_W, height=self.RESULT_H,
            bg="#0a0e14", highlightthickness=1, highlightbackground=BORDER,
            cursor="crosshair")
        self._result_canvas.grid(row=1, column=1, sticky="nsew")
        self._result_canvas.create_text(
            self.RESULT_W//2, self.RESULT_H//2,
            text="Captured image appears here", fill=MUTED, font=FONT_MONO, tags="placeholder")
        self._result_canvas.bind("<ButtonPress-1>",   self._roi_mouse_down)
        self._result_canvas.bind("<B1-Motion>",       self._roi_mouse_move)
        self._result_canvas.bind("<ButtonRelease-1>", self._roi_mouse_up)

        # Sidebar
        side = tk.Frame(body, bg=PANEL_BG, width=245,
                        highlightthickness=1, highlightbackground=BORDER)
        side.grid(row=0, column=1, sticky="ns", padx=(8,0))
        side.columnconfigure(0, weight=1)
        side.grid_propagate(False)
        self._build_sidebar(side)

        # Log
        log_frame = tk.Frame(self, bg=PANEL_BG,
                             highlightthickness=1, highlightbackground=BORDER)
        log_frame.grid(row=2, column=0, sticky="ew", padx=12, pady=(0,10))
        log_frame.columnconfigure(0, weight=1)

        tk.Label(log_frame, text="ACTIVITY LOG", font=FONT_TITLE,
                 bg=PANEL_BG, fg=MUTED, padx=8, pady=4
                 ).grid(row=0, column=0, sticky="w")
        self._log_text = tk.Text(
            log_frame, height=7, bg="#0d1117", fg=TEXT_FG, font=FONT_MONO,
            relief="flat", insertbackground=ACCENT, selectbackground=BORDER,
            wrap="word", state="disabled")
        self._log_text.grid(row=1, column=0, sticky="ew", padx=6, pady=(0,6))
        sb = ttk.Scrollbar(log_frame, command=self._log_text.yview)
        sb.grid(row=1, column=1, sticky="ns", pady=(0,6))
        self._log_text["yscrollcommand"] = sb.set

    def _build_sidebar(self, parent):
        pad = {"padx": 12, "pady": 5}
        row = 0

        def section(text, r):
            tk.Label(parent, text=text, font=("Segoe UI", 9, "bold"),
                     bg=PANEL_BG, fg=MUTED
                     ).grid(row=r, column=0, sticky="w", padx=12, pady=(12,2))

        def sep(r):
            tk.Frame(parent, bg=BORDER, height=1
                     ).grid(row=r, column=0, sticky="ew", padx=10, pady=6)

        section("CONTROLS", row); row += 1
        self._btn_capture = self._make_btn(
            parent, "📷  Capture Image", self._action_capture, row=row, **pad)
        row += 1

        sep(row); row += 1
        section("CALIBRATION", row); row += 1

        self._btn_set_roi = self._make_btn(
            parent, "✏  Draw ROI", self._action_set_roi, row=row, **pad)
        row += 1

        self._btn_edit_roi = self._make_btn(
            parent, "🔢  Edit ROI Coords", self._action_edit_roi, row=row, **pad)
        row += 1

        self._btn_calibrate = self._make_btn(
            parent, "⚙  Calibrate (Set Reference)", self._action_calibrate,
            row=row, **pad)
        row += 1

        sep(row); row += 1
        section("INSPECTION", row); row += 1

        self._btn_inspect = self._make_btn(
            parent, "🔍  Inspect Captured Image", self._action_inspect,
            row=row, **pad)
        row += 1

        self._btn_live_inspect = self._make_btn(
            parent, "▶  Start Live Inspect", self._action_toggle_live_inspect,
            row=row, **pad)
        self._btn_live_inspect.configure(fg=GREEN)
        row += 1

        sep(row); row += 1
        section("CURRENT ROI", row); row += 1
        self._roi_var = tk.StringVar()
        self._update_roi_label()
        tk.Label(parent, textvariable=self._roi_var, font=("Courier New", 9),
                 bg=PANEL_BG, fg=TEXT_FG, justify="left"
                 ).grid(row=row, column=0, sticky="w", padx=12, pady=(0,6))
        row += 1

        sep(row); row += 1
        section("BASELINE", row); row += 1
        self._baseline_var = tk.StringVar(value="Not loaded")
        self._baseline_lbl = tk.Label(parent, textvariable=self._baseline_var,
                                      font=FONT_MONO, bg=PANEL_BG, fg=ORANGE,
                                      wraplength=200)
        self._baseline_lbl.grid(row=row, column=0, sticky="w", padx=12, pady=(0,6))
        row += 1

        self._result_var = tk.StringVar(value="")
        self._result_banner = tk.Label(
            parent, textvariable=self._result_var,
            font=("Segoe UI", 20, "bold"), bg=PANEL_BG, fg=MUTED,
            relief="flat", pady=10)
        self._result_banner.grid(row=row, column=0, sticky="ew", padx=12, pady=4)
        row += 1

        self._detail_var = tk.StringVar(value="")
        tk.Label(parent, textvariable=self._detail_var,
                 font=("Courier New", 9), bg=PANEL_BG, fg=TEXT_FG,
                 wraplength=210, justify="left"
                 ).grid(row=row, column=0, sticky="w", padx=12)
        row += 1

        sep(row); row += 1
        section("BUFFER STATUS", row); row += 1
        self._buffer_var = tk.StringVar(value="—")
        tk.Label(parent, textvariable=self._buffer_var,
                 font=("Courier New", 9), bg=PANEL_BG, fg=MUTED
                 ).grid(row=row, column=0, sticky="w", padx=12, pady=(0,8))

    def _make_btn(self, parent, text, cmd, row, **grid_kw):
        btn = tk.Button(
            parent, text=text, command=cmd,
            bg=BTN_BG, fg=TEXT_FG, activebackground=BTN_HOV,
            activeforeground=TEXT_FG, relief="flat", bd=0,
            padx=10, pady=7, font=("Segoe UI", 10), anchor="w",
            width=24, cursor="hand2")
        btn.grid(row=row, column=0, sticky="ew", **grid_kw)
        btn.bind("<Enter>", lambda e: btn.configure(bg=BTN_HOV))
        btn.bind("<Leave>", lambda e: btn.configure(bg=BTN_BG))
        return btn

    # ── LOGGING ───────────────────────────────────────────────────
    def _log(self, msg):
        ts   = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}]  {msg}\n"
        self._log_text.configure(state="normal")
        self._log_text.insert("end", line)
        self._log_text.see("end")
        self._log_text.configure(state="disabled")

    def _set_status(self, msg):
        self._status_var.set(msg)

    # ── BASELINE ──────────────────────────────────────────────────
    def _try_load_baseline(self):
        try:
            self.baseline_profile = load_baseline()
            self._baseline_var.set("✓ Loaded")
            self._baseline_lbl.configure(fg=GREEN)
            self._log("Baseline loaded from baseline.json")
        except FileNotFoundError:
            self._baseline_var.set("Not calibrated")
            self._baseline_lbl.configure(fg=ORANGE)

    def _update_roi_label(self):
        self._roi_var.set(
            f"X: {CONFIG['ROI_X_START']} → {CONFIG['ROI_X_END']}\n"
            f"Y: {CONFIG['ROI_Y_START']} → {CONFIG['ROI_Y_END']}"
        )

    # ── CAMERA START ──────────────────────────────────────────────
    def _start_camera(self):
        try:
            if CAMERA_BACKEND == 'picamera2':
                cam = Picamera2()
                cfg = cam.create_still_configuration(
                    main={"size": CONFIG["CAPTURE_RESOLUTION"], "format": "RGB888"},
                    controls={
                        "AfMode": 0, "LensPosition": CONFIG["LENS_POSITION"],
                        "ExposureTime": CONFIG["EXPOSURE_TIME"],
                        "AnalogueGain": 1.0, "AwbEnable": False,
                        "ColourGains": (1.5, 1.5),
                    })
                cam.configure(cfg)
                cam.start()
                time.sleep(1.5)
                self.cam = cam
            else:
                self.cam = WebcamCapture(CONFIG["WEBCAM_INDEX"])

            self._preview_running = True
            self._preview_thread  = threading.Thread(
                target=self._preview_loop, daemon=True)
            self._preview_thread.start()
            self._log(f"Camera started [{CAMERA_BACKEND}]. Live preview active.")
            self._set_status(f"Camera live [{CAMERA_BACKEND}]")

        except Exception as e:
            self._log(f"⚠  Camera error: {e}")
            self._set_status("No camera")

    # ── PREVIEW LOOP ──────────────────────────────────────────────
    def _preview_loop(self):
        while self._preview_running:
            try:
                # Capture RGB frame
                rgb  = self.cam.capture_array()          # always RGB
                bgr  = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                bgr  = normalize_image(bgr)
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                bgr_ov = bgr.copy()

                # ── Multi-frame calibration ────────────────────────
                if self._calibrating:
                    try:
                        raw_profile = get_edge_profile(gray)
                        self._baseline_collector.add_frame(raw_profile)
                        if self._calib_rgb_sample is None:
                            self._calib_rgb_sample = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                        pct = int(self._baseline_collector.progress * 100)
                        self.after(0, lambda p=pct: self._buffer_var.set(
                            f"Calibrating… {p}%"))
                        if self._baseline_collector.complete:
                            self.after(0, self._finish_calibration)
                    except Exception:
                        pass

                # ── Live inspection ────────────────────────────────
                elif self._live_inspect_active and self.baseline_profile is not None:
                    self._live_frame_count += 1
                    try:
                        raw_profile = get_edge_profile(gray)
                        self._temporal_smoother.push(raw_profile)
                    except Exception:
                        pass

                    fill_pct = int(self._temporal_smoother.fill_fraction * 100)
                    self.after(0, lambda p=fill_pct: self._buffer_var.set(
                        f"Warming… {p}%" if p < 100 else "Buffer ready ✓"))

                    if (self._live_frame_count >= CONFIG["LIVE_INSPECT_EVERY"]
                            and self._temporal_smoother.ready):
                        self._live_frame_count = 0
                        try:
                            smooth = self._temporal_smoother.get_smoothed()
                            passed, fail_regions, diff, diff_mm, max_abs_mm = analyze_uplift(
                                smooth, self.baseline_profile)
                            self._live_result = "PASS" if passed else "FAIL"
                            if passed:
                                self._live_result_detail = f"max {max_abs_mm:.3f} mm"
                            else:
                                dirs = [("↑" if r["direction"] == "lifted" else "↓")
                                        for r in fail_regions]
                                self._live_result_detail = (
                                    f"max {max_abs_mm:.3f} mm  "
                                    + "  ".join(
                                        f"{d}{r['max_uplift_mm']:.2f}mm"
                                        for d, r in zip(dirs, fail_regions)))
                            self.after(0, self._update_live_result_banner)
                        except Exception as ex:
                            self._live_result = "ERR"
                            self._live_result_detail = str(ex)

                # ── ROI overlay ────────────────────────────────────
                roi_color = (0, 165, 255)
                if self._calibrating:
                    roi_color = (0, 165, 255)
                elif self._live_inspect_active:
                    roi_color = (0, 220, 0) if self._live_result == "PASS" else (0, 0, 255)

                cv2.rectangle(bgr_ov,
                              (CONFIG["ROI_X_START"], CONFIG["ROI_Y_START"]),
                              (CONFIG["ROI_X_END"],   CONFIG["ROI_Y_END"]),
                              roi_color, 3)

                if self._calibrating:
                    pct = int(self._baseline_collector.progress * 100)
                    cv2.putText(bgr_ov, f"CALIBRATING {pct}%",
                                (CONFIG["ROI_X_START"],
                                 max(CONFIG["ROI_Y_START"] - 18, 60)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 165, 255), 3)

                elif self._live_inspect_active and self._live_result:
                    vcolor = (0, 220, 0) if self._live_result == "PASS" else (0, 0, 255)
                    ty = max(CONFIG["ROI_Y_START"] - 18, 60)
                    cv2.putText(bgr_ov, self._live_result,
                                (CONFIG["ROI_X_START"], ty),
                                cv2.FONT_HERSHEY_SIMPLEX, 2.0, vcolor, 4)
                    cv2.putText(bgr_ov, self._live_result_detail,
                                (CONFIG["ROI_X_START"], ty + 48),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, vcolor, 2)

                with self._live_lock:
                    self._live_frame_bgr = bgr_ov
                self.after(0, self._refresh_preview)

            except Exception:
                pass
            time.sleep(0.04)   # ~25 fps max

    def _refresh_preview(self):
        with self._live_lock:
            frame = self._live_frame_bgr
        if frame is None:
            return
        photo = pil_from_bgr(frame, self.PREVIEW_W, self.PREVIEW_H)
        self._live_canvas.delete("all")
        cw = self._live_canvas.winfo_width()  or self.PREVIEW_W
        ch = self._live_canvas.winfo_height() or self.PREVIEW_H
        self._live_canvas.create_image(cw//2, ch//2, anchor="center",
                                       image=photo, tags="frame")
        self._live_canvas._photo = photo

    # ── FINISH CALIBRATION ────────────────────────────────────────
    def _finish_calibration(self):
        if not self._baseline_collector.complete:
            return
        self._calibrating = False
        try:
            profile = self._baseline_collector.get_baseline()
            rgb_img = (self._calib_rgb_sample
                       if self._calib_rgb_sample is not None
                       else np.zeros((100, 100, 3), dtype=np.uint8))
            save_baseline(profile, rgb_img)
            self.baseline_profile = profile
            n = CONFIG["BASELINE_FRAMES"]
            self._baseline_var.set(f"✓ Calibrated ({n}f)")
            self._baseline_lbl.configure(fg=GREEN)
            self._temporal_smoother.reset()   # force live to re-warm
            self._btn_calibrate.configure(
                text="⚙  Calibrate (Set Reference)", state="normal")
            self._log(
                f"Calibration done. {n}-frame median baseline. "
                f"Median Y = {np.median(profile):.1f} px")
            self._set_status("Calibrated")
            self._result_var.set("")
            self._detail_var.set("")
            self._buffer_var.set("Calibration complete ✓")
        except Exception as e:
            self._log(f"⚠  Calibration error: {e}")
            self._btn_calibrate.configure(state="normal")

    # ── LIVE INSPECT TOGGLE ───────────────────────────────────────
    def _action_toggle_live_inspect(self):
        if not CAMERA_AVAILABLE or self.cam is None:
            messagebox.showinfo("No Camera",
                                "Live Inspect requires a connected camera.")
            return
        if self.baseline_profile is None:
            messagebox.showinfo("No Baseline",
                                "Calibrate first before starting Live Inspect.")
            return

        self._live_inspect_active = not self._live_inspect_active

        if self._live_inspect_active:
            self._live_result        = None
            self._live_result_detail = ""
            self._live_frame_count   = 0
            self._temporal_smoother.reset()
            self._btn_live_inspect.configure(text="⏹  Stop Live Inspect", fg=RED)
            self._log(
                f"Live Inspect started — {CONFIG['TEMPORAL_FRAMES']}-frame median, "
                f"detrend order={CONFIG['DETREND_ORDER']}, "
                f"threshold={CONFIG['MAX_MM_THRESHOLD']} mm.")
            self._set_status("Live Inspect ON")
            self._result_var.set("⏳  Warming…")
            self._result_banner.configure(fg=ORANGE)
            self._detail_var.set(f"Collecting {CONFIG['TEMPORAL_FRAMES']} frames…")
        else:
            self._btn_live_inspect.configure(text="▶  Start Live Inspect", fg=GREEN)
            self._live_result        = None
            self._live_result_detail = ""
            self._result_var.set("")
            self._detail_var.set("")
            self._buffer_var.set("—")
            self._temporal_smoother.reset()
            self._log("Live Inspect stopped.")
            self._set_status("Live Inspect OFF")

    def _update_live_result_banner(self):
        if not self._live_inspect_active or self._live_result is None:
            return
        if self._live_result == "PASS":
            self._result_var.set("✅  PASS")
            self._result_banner.configure(fg=GREEN)
        else:
            self._result_var.set("❌  FAIL")
            self._result_banner.configure(fg=RED)
        self._detail_var.set(self._live_result_detail)

    # ── CAPTURE ───────────────────────────────────────────────────
    def _action_capture(self):
        if self.cam:
            self._capture_from_camera()
        else:
            self._capture_from_file()

    def _capture_from_camera(self):
        self._set_status("Capturing…")
        self._log("Capturing from camera…")
        try:
            rgb  = self.cam.capture_array()
            bgr  = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            bgr  = normalize_image(bgr)
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            self.last_rgb  = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            self.last_gray = gray
            self._show_captured(bgr)
            self._log(f"Captured. Size: {bgr.shape[1]}×{bgr.shape[0]} px")
            self._set_status("Image captured")
        except Exception as e:
            self._log(f"⚠  Capture error: {e}")
            self._set_status("Capture failed")

    def _capture_from_file(self):
        from tkinter import filedialog
        path = filedialog.askopenfilename(
            title="Select PCB image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if not path:
            return
        try:
            bgr  = cv2.imread(path)
            bgr  = normalize_image(bgr)
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            self.last_rgb  = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            self.last_gray = gray
            self._show_captured(bgr)
            self._log(f"Loaded: {os.path.basename(path)}")
            self._set_status("File loaded")
        except Exception as e:
            self._log(f"⚠  File load error: {e}")

    def _show_captured(self, bgr):
        cw = self._result_canvas.winfo_width()  or self.RESULT_W
        ch = self._result_canvas.winfo_height() or self.RESULT_H
        h, w = bgr.shape[:2]
        scale = min(cw/w, ch/h, 1.0)
        nw, nh = int(w*scale), int(h*scale)
        self._result_scale  = scale
        self._result_offset = ((cw-nw)//2, (ch-nh)//2)
        photo = pil_from_bgr(bgr, cw, ch)
        self._result_canvas.delete("all")
        self._result_canvas.create_image(cw//2, ch//2, anchor="center",
                                         image=photo, tags="captured")
        self._result_canvas._photo = photo
        self._draw_roi_on_canvas()

    def _draw_roi_on_canvas(self):
        self._result_canvas.delete("roi_box")
        sc = self._result_scale
        ox, oy = self._result_offset
        x1 = int(CONFIG["ROI_X_START"] * sc) + ox
        y1 = int(CONFIG["ROI_Y_START"] * sc) + oy
        x2 = int(CONFIG["ROI_X_END"]   * sc) + ox
        y2 = int(CONFIG["ROI_Y_END"]   * sc) + oy
        self._result_canvas.create_rectangle(
            x1, y1, x2, y2, outline=ORANGE, width=2, tags="roi_box", dash=(6,4))

    # ── ROI DRAWING ───────────────────────────────────────────────
    def _action_set_roi(self):
        if self.last_rgb is None:
            messagebox.showinfo("No Image",
                                "Capture or load an image first, then draw ROI.")
            return
        self._roi_mode = True
        self._result_canvas.configure(cursor="crosshair")
        self._log("ROI mode ON — click and drag on the result image.")
        self._set_status("Draw ROI on captured image…")

    def _roi_mouse_down(self, event):
        if not self._roi_mode:
            return
        self._roi_drawing = True
        self._roi_start   = (event.x, event.y)
        if self._roi_rect_id:
            self._result_canvas.delete(self._roi_rect_id)

    def _roi_mouse_move(self, event):
        if not self._roi_mode or not self._roi_drawing:
            return
        if self._roi_rect_id:
            self._result_canvas.delete(self._roi_rect_id)
        x0, y0 = self._roi_start
        self._roi_rect_id = self._result_canvas.create_rectangle(
            x0, y0, event.x, event.y, outline=GREEN, width=2, dash=(4,3))

    def _roi_mouse_up(self, event):
        if not self._roi_mode or not self._roi_drawing:
            return
        self._roi_drawing = False
        self._roi_mode    = False
        self._result_canvas.configure(cursor="crosshair")
        x0, y0 = self._roi_start
        x1, y1 = event.x, event.y
        sc = self._result_scale
        ox, oy = self._result_offset
        img_x0 = int((min(x0,x1) - ox) / sc)
        img_y0 = int((min(y0,y1) - oy) / sc)
        img_x1 = int((max(x0,x1) - ox) / sc)
        img_y1 = int((max(y0,y1) - oy) / sc)
        H, W = self.last_rgb.shape[:2]
        img_x0 = max(0, min(img_x0, W))
        img_y0 = max(0, min(img_y0, H))
        img_x1 = max(0, min(img_x1, W))
        img_y1 = max(0, min(img_y1, H))
        if abs(img_x1-img_x0) < 20 or abs(img_y1-img_y0) < 5:
            self._log("⚠  ROI too small — try again.")
            self._set_status("ROI too small")
            return
        CONFIG["ROI_X_START"] = img_x0
        CONFIG["ROI_Y_START"] = img_y0
        CONFIG["ROI_X_END"]   = img_x1
        CONFIG["ROI_Y_END"]   = img_y1
        save_roi_config()
        self._update_roi_label()
        self._draw_roi_on_canvas()
        self._log(f"ROI saved: X {img_x0}→{img_x1}  Y {img_y0}→{img_y1}")
        self._set_status("ROI updated")

    # ── EDIT ROI DIALOG ───────────────────────────────────────────
    def _action_edit_roi(self):
        def on_apply():
            self._update_roi_label()
            if self.last_rgb is not None:
                bgr = cv2.cvtColor(self.last_rgb, cv2.COLOR_RGB2BGR)
                self._show_captured(bgr)
            self._log(
                f"ROI updated: X {CONFIG['ROI_X_START']}→{CONFIG['ROI_X_END']}  "
                f"Y {CONFIG['ROI_Y_START']}→{CONFIG['ROI_Y_END']}")
            self._set_status("ROI updated")
        ROIEditDialog(self, on_apply)

    # ── CALIBRATE ─────────────────────────────────────────────────
    def _action_calibrate(self):
        if not CAMERA_AVAILABLE or self.cam is None:
            self._calibrate_from_file()
            return

        ans = messagebox.askyesno(
            "Calibrate",
            f"This will capture {CONFIG['BASELINE_FRAMES']} frames to build a stable baseline.\n"
            "Ensure the jig is in place with NO PCB on it.\n\n"
            "Keep everything still during calibration.\n\nContinue?")
        if not ans:
            return

        self._baseline_collector.reset()
        self._calib_rgb_sample = None
        self._calibrating      = True
        self._btn_calibrate.configure(state="disabled", text="⚙  Calibrating…")
        self._buffer_var.set("Calibrating… 0%")
        self._log(f"Calibration started — collecting {CONFIG['BASELINE_FRAMES']} frames…")
        self._set_status("Calibrating…")

    def _calibrate_from_file(self):
        """
        Single-image fallback (no camera).
        We simulate multi-frame noise reduction by running the profile
        extraction 5 times on the same image (since the image is static,
        the median will equal the single result but the code path is the
        same).
        """
        if self.last_rgb is None or self.last_gray is None:
            messagebox.showinfo("No Image",
                                "Load the BARE JIG image first, then calibrate.")
            return
        ans = messagebox.askyesno(
            "Calibrate (file mode)",
            "No camera — will calibrate from the currently loaded image.\n\nContinue?")
        if not ans:
            return
        try:
            # Repeat profile extraction to simulate multi-frame median
            profiles = [get_edge_profile(self.last_gray) for _ in range(5)]
            profile  = np.median(np.stack(profiles, axis=0), axis=0).astype(np.float32)
            save_baseline(profile, self.last_rgb)
            self.baseline_profile = profile
            self._baseline_var.set("✓ Calibrated (file)")
            self._baseline_lbl.configure(fg=ORANGE)
            self._temporal_smoother.reset()
            self._log(
                f"File calibration done. Baseline median Y = {np.median(profile):.1f} px")
            self._set_status("Calibrated (file)")
        except Exception as e:
            self._log(f"⚠  Calibration error: {e}")

    # ── INSPECT ───────────────────────────────────────────────────
    def _action_inspect(self):
        if self.last_gray is None:
            messagebox.showinfo("No Image", "Capture or load a PCB image first.")
            return
        if self.baseline_profile is None:
            messagebox.showinfo("No Baseline", "No baseline loaded. Run Calibrate first.")
            return

        self._set_status("Inspecting…")
        self._log("Running inspection…")
        try:
            current_profile = get_edge_profile(self.last_gray)
            passed, fail_regions, diff, diff_mm, max_abs_mm = analyze_uplift(
                current_profile, self.baseline_profile)

            bgr = cv2.cvtColor(self.last_rgb, cv2.COLOR_RGB2BGR)
            ann = build_annotated_image(
                bgr, current_profile, self.baseline_profile,
                diff, fail_regions, passed)

            cw = self._result_canvas.winfo_width()  or self.RESULT_W
            ch = self._result_canvas.winfo_height() or self.RESULT_H
            photo = pil_from_bgr(ann, cw, ch)
            self._result_canvas.delete("all")
            self._result_canvas.create_image(cw//2, ch//2, anchor="center",
                                             image=photo, tags="result")
            self._result_canvas._photo = photo

            ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
            label    = "PASS" if passed else "FAIL"
            os.makedirs(CONFIG["LOG_DIR"], exist_ok=True)
            img_path = os.path.join(CONFIG["LOG_DIR"], f"{ts}_{label}.jpg")
            cv2.imwrite(img_path, ann)
            write_log(ts, "UI", passed, max_abs_mm, fail_regions, img_path)

            if passed:
                self._result_var.set("✅  PASS")
                self._result_banner.configure(fg=GREEN)
                self._detail_var.set(
                    f"Max deviation: {max_abs_mm:.3f} mm\n"
                    f"Threshold: {CONFIG['MAX_MM_THRESHOLD']} mm")
            else:
                self._result_var.set("❌  FAIL")
                self._result_banner.configure(fg=RED)
                details = [f"Max deviation: {max_abs_mm:.3f} mm"]
                for i, r in enumerate(fail_regions, 1):
                    d = "↑ lifted" if r["direction"] == "lifted" else "↓ sunken"
                    details.append(
                        f"  {i}. {d}  {r['max_uplift_mm']:.2f}mm "
                        f"@ {r['x_start_mm']:.0f}–{r['x_end_mm']:.0f}mm")
                self._detail_var.set("\n".join(details))

            self._log(f"Result: {label}  |  max={max_abs_mm:.3f}mm  |  → {img_path}")
            self._set_status(label)

        except Exception as e:
            self._log(f"⚠  Inspection error: {e}")
            self._set_status("Error")

    # ── CLOSE ─────────────────────────────────────────────────────
    def _on_close(self):
        self._preview_running    = False
        self._calibrating        = False
        self._live_inspect_active = False
        if self.cam:
            try:
                self.cam.stop()
            except Exception:
                pass
        self.destroy()


# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────
if __name__ == "__main__":
    app = PCBApp()
    app.mainloop()
