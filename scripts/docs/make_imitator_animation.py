#!/usr/bin/env python
"""Render the multi-scene Imitator explainer GIF used in the README.

Three scenes: keypoint capture -> internal processing -> decoded output.

Scene 1 plays the real keypoints of clip 1660 ("Hambriento") and scene 3 shows
the checkpoint's real output for that same clip, so both ends of the figure are
actual data. Scene 2 is a diagram: the graph pulses and the attention
weights are drawn, not recorded. The prediction comes from the teacher-forced
evaluation, where the CIF alphas and the token count are taken from the target
(train_temporal_v126.py:668-688); free-running, that run scored
pred_raw_top1=0.369 / pred_exact=0.074 against teacher_top1=0.901.

ponytail: pure Pillow + numpy on purpose. The active interpreter has no
matplotlib/h5py/torch, so the keypoints are read from the committed
docs/clip_keypoints.npz rather than from the HDF5 dataset.

    python scripts/docs/make_imitator_animation.py --out docs/imitator.gif
    python scripts/docs/make_imitator_animation.py --selftest
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# --- canvas -----------------------------------------------------------------

W, H = 960, 420
SS = 2  # supersampling: draw big, downscale -> free antialiasing
BG = (11, 15, 25)

# Monocromo Tecnológico / Cool Tech (DeepMind / NeurIPS publication palette)
CYAN = (56, 189, 248)     # keypoints / ST-GCN / nodes (111)
SKY = (147, 197, 253)     # pooled node features (512)
TEAL = (45, 212, 191)     # frame sequence / transformer (T) / left hand
INDIGO = (165, 180, 252)  # token queries (20) / right hand / vocab
WHITE = (248, 250, 252)   # projection / Gemma embedding (3072) / final text
DIM_BLUE = (45, 60, 85)   # facial keypoints (subtle slate, zero crowding)
TEXT = (226, 232, 240)    # crisp readable text
MUTED = (148, 163, 184)   # secondary text / brackets
STROKE = (51, 65, 85)     # elegant slate panel borders

BLUE = CYAN
ORANGE = CYAN
GREEN = TEAL
PURPLE = INDIGO

# Real prediction for the same clip the keypoints come from: 1660, gloss
# "Hambriento", predicted == target. Picked over the other exact-match rows
# because both hands are cleanly detected in its keypoints.
# outputs/v126_temporal/diag_imitator_A2_predlen_optunaStable_20260625_194336/predictions.jsonl
TOKEN_IDS = [236754, 1525, 604, 14246]
SUBTOKENS = ["<bos>", " ham", "brien", "to"]
PRED_TEXT = "hambriento"
GLOSS = "Hambriento"

FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")


def font(size: int, mono: bool = False, bold: bool = False):
    name = "DejaVuSansMono" if mono else "DejaVuSans"
    if bold:
        name += "-Bold"
    path = FONT_DIR / f"{name}.ttf"
    if path.exists():
        return ImageFont.truetype(str(path), size * SS)
    return ImageFont.load_default()


# --- skeleton ---------------------------------------------------------------
# Real keypoints: docs/clip_keypoints.npz holds the 111 model-input keypoints of
# clip 1660 ("Hambriento"), exported by scripts/docs/export_clip_keypoints.py.
# Layout after remove_keypoints (data_augmentation.py:142-151):
#   0-6 pose (OpenPose nose, neck, R shoulder/elbow/wrist, L shoulder/elbow)
#   7-70 face (64)   71-90 left hand (20)   91-110 right hand (20)
CLIP_NPZ = Path(__file__).resolve().parents[2] / "docs" / "clip_keypoints.npz"

LH, RH = 71, 91
POSE_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6)]
# The 20-point hand blocks are 5 fingers of 4 joints; the OpenPose hand root is
# not part of the 111, so the fingers are tied together through the palm chain.
HAND_EDGES = [(f * 4 + j, f * 4 + j + 1) for f in range(5) for j in range(3)]
HAND_EDGES += [(f * 4, (f + 1) * 4) for f in range(4)]
# The left wrist is not in the 111 either, so the left hand hangs off the elbow.
ATTACH = [(6, LH), (4, RH)]

REGIONS = ((slice(0, 7), BLUE), (slice(7, 71), DIM_BLUE),
           (slice(LH, LH + 20), GREEN), (slice(RH, RH + 20), PURPLE))

_CLIP = None


def clip() -> dict:
    """Load the exported clip once: keypoints [T, 111, 2] in [0, 1], y down."""
    global _CLIP
    if _CLIP is None:
        d = np.load(CLIP_NPZ)
        _CLIP = {"kp": d["keypoints"].astype(np.float32),
                 "gloss": str(d["gloss"]), "clip_id": str(d["clip_id"])}
    return _CLIP


def frame_points(t: float, cx: float, cy: float, scale: float) -> np.ndarray:
    """Keypoints of the clip at position t in [0, 1], mapped to pixels."""
    kp = clip()["kp"]
    i = int(round(max(0.0, min(1.0, t)) * (len(kp) - 1)))
    return (kp[i] - 0.5) * scale + np.array([cx, cy], dtype=np.float32)


def bone_segments(pts: np.ndarray, max_len: float = 1e9):
    """Bones for one frame, skipping implausibly long ones.

    Every one of the 111 points is still drawn; the length guard only hides
    bones to mis-detected joints (some clips have a noisy non-dominant hand),
    which would otherwise draw lines across the whole panel.
    """
    edges = list(POSE_EDGES) + ATTACH
    edges += [(LH + a, LH + b) for a, b in HAND_EDGES]
    edges += [(RH + a, RH + b) for a, b in HAND_EDGES]
    out = []
    for a, b in edges:
        if np.hypot(*(pts[a] - pts[b])) <= max_len:
            out.append([tuple(pts[a]), tuple(pts[b])])
    return out


# --- drawing helpers --------------------------------------------------------

def blend(c, a: float, bg=BG):
    a = max(0.0, min(1.0, a))
    return tuple(int(bg[i] + (c[i] - bg[i]) * a) for i in range(3))


def s(v):
    return v * SS


def line(d, pts, color, width=1.4, alpha=1.0):
    if alpha <= 0.01 or len(pts) < 2:
        return
    d.line([(s(x), s(y)) for x, y in pts], fill=blend(color, alpha),
           width=max(1, int(width * SS)), joint="curve")


def dot(d, p, r, color, alpha=1.0):
    if alpha <= 0.01:
        return
    x, y = s(p[0]), s(p[1])
    rr = r * SS
    d.ellipse([x - rr, y - rr, x + rr, y + rr], fill=blend(color, alpha))


def box(d, xy, color=STROKE, alpha=1.0, radius=8, fill=None):
    x0, y0, x1, y1 = [s(v) for v in xy]
    d.rounded_rectangle([x0, y0, x1, y1], radius=radius * SS,
                        outline=blend(color, alpha), width=int(1.5 * SS),
                        fill=fill)


def text(d, p, msg, f, color=TEXT, alpha=1.0, anchor="la"):
    if alpha <= 0.01:
        return
    d.text((s(p[0]), s(p[1])), msg, font=f, fill=blend(color, alpha), anchor=anchor)


def new_frame():
    img = Image.new("RGB", (W * SS, H * SS), BG)
    return img, ImageDraw.Draw(img)


class Fonts:
    def __init__(self):
        self.title = font(15, bold=True)
        self.label = font(12, bold=True)
        self.small = font(10)
        self.tiny = font(9)
        self.badge = font(10, bold=True)
        self.mono = font(12, mono=True)
        self.mono_small = font(11, mono=True)
        self.mono_card_id = font(16, mono=True, bold=True)
        self.mono_card_sub = font(13, mono=True, bold=True)
        self.mono_big = font(20, mono=True)
        self.big = font(24, bold=True)
        self.hero = font(34, bold=True)


FT = None


# --- scenes -----------------------------------------------------------------

SCENES = ("1 · Capture", "2 · Processing", "3 · Output")


def breadcrumb(d, idx: int, p: float):
    y = 396
    seg = 240
    x0 = (W - seg * 3) / 2
    for i, name in enumerate(SCENES):
        x = x0 + i * seg
        active = i == idx
        line(d, [(x, y), (x + seg - 16, y)], STROKE, 2.0, 0.25)
        if i < idx:
            line(d, [(x, y), (x + seg - 16, y)], BLUE, 2.0, 0.5)
        if active:
            line(d, [(x, y), (x + (seg - 16) * p, y)], BLUE, 2.5, 1.0)
        text(d, (x + (seg - 16) / 2, y + 8), name, FT.small,
             WHITE if active else MUTED, 1.0 if active else 0.55, anchor="ma")


def draw_tensor_badge(d, cx: float, cy: float, shape_str: str, color=MUTED, alpha: float = 1.0, glow: float = 0.0):
    if alpha <= 0.01:
        return
    f = FT.mono_small
    txt_w = f.getlength(shape_str) / SS
    bb = [v / SS for v in f.getbbox(shape_str, anchor="la")]
    txt_h = bb[3] - bb[1]

    pad_x, pad_y = 8, 4
    x0, y0 = cx - txt_w / 2 - pad_x, cy - txt_h / 2 - pad_y
    x1, y1 = cx + txt_w / 2 + pad_x, cy + txt_h / 2 + pad_y

    fill_alpha = alpha * (0.16 + 0.22 * glow)
    stroke_alpha = alpha * (0.45 + 0.45 * glow)

    d.rounded_rectangle(
        [s(x0), s(y0), s(x1), s(y1)],
        radius=s(4),
        fill=blend(color, fill_alpha),
        outline=blend(color if glow > 0.3 else STROKE, stroke_alpha),
        width=max(1, int(1.2 * SS)),
    )
    text_col = WHITE if glow > 0.3 else TEXT
    text(d, (cx, cy), shape_str, f, text_col, alpha, anchor="mm")


def draw_pill(d, cx: float, cy: float, label: str, color, alpha: float = 1.0):
    if alpha <= 0.01:
        return
    f = FT.badge
    txt_w = f.getlength(label) / SS
    bb = [v / SS for v in f.getbbox(label, anchor="la")]
    txt_h = bb[3] - bb[1]
    pad_x, pad_y = 10, 4
    x0, y0 = cx - txt_w / 2 - pad_x, cy - txt_h / 2 - pad_y
    x1, y1 = cx + txt_w / 2 + pad_x, cy + txt_h / 2 + pad_y
    d.rounded_rectangle(
        [s(x0), s(y0), s(x1), s(y1)],
        radius=s(11),
        fill=blend(color, alpha * 0.20),
        outline=blend(color, alpha * 0.80),
        width=max(1, int(1.0 * SS)),
    )
    text(d, (cx, cy), label, f, color, alpha, anchor="mm")


def scene_capture(p: float):
    img, d = new_frame()
    c = clip()
    text(d, (W / 2, 26), "From video to keypoints", FT.title, WHITE, anchor="ma")

    # Motion plays over first 68%, then holds final pose for cognitive dwell time
    p_motion = min(1.0, p / 0.68)

    # left: the clip itself, as raw tracked joints
    box(d, (70, 60, 360, 350))
    text(d, (78, 44), f"clip {c['clip_id']} · “{c['gloss']}”  (134 tracked joints)",
         FT.small, MUTED)
    pts = frame_points(p_motion, 215, 195, 190)

    fade = min(1.0, max(0.0, (p_motion - 0.35) / 0.3))   # bones dissolve, dots remain
    for b in bone_segments(pts):
        line(d, b, CYAN, 1.6, 0.85 * (1 - 0.75 * fade))
    for q in pts:
        dot(d, q, 1.4 + 0.8 * fade, CYAN, 0.35 + 0.6 * fade)

    # arrow
    line(d, [(375, 205), (440, 205)], STROKE, 1.5)
    line(d, [(432, 200), (440, 205), (432, 210)], STROKE, 1.5)
    text(d, (407, 182), "filter & norm", FT.small, TEXT, anchor="ma")
    text(d, (407, 218), "134 → 111 joints", FT.tiny, MUTED, anchor="ma")

    # right: model input, 111 normalized joints colored by semantic region
    box(d, (455, 60, 890, 350))
    text(d, (463, 44), "model input · 111 normalized joints", FT.small, MUTED)
    pts2 = frame_points(p_motion, 540, 195, 190)
    for sl, col in REGIONS:
        r_dot = 1.15 if col == DIM_BLUE else (2.0 if col == CYAN else 1.8)
        alpha_dot = 0.65 if col == DIM_BLUE else 0.95
        for q in pts2[sl]:
            dot(d, q, r_dot, col, alpha_dot)

    rows = [
        ("pose", 7, CYAN),
        ("face", 64, DIM_BLUE),
        ("left hand", 20, TEAL),
        ("right hand", 20, INDIGO),
    ]
    y = 118
    for i, (name, n, col) in enumerate(rows):
        a = min(1.0, max(0.0, (p_motion - 0.12 * i) / 0.16))
        dot(d, (716, y - 4), 3.5, col, a)
        text(d, (730, y - 11), name, FT.small, TEXT, a)
        text(d, (874, y - 11), f"{n}", FT.mono, col, a, anchor="ra")
        y += 28

    line(d, [(716, 228), (874, 228)], STROKE, 1.2, 0.5)
    text(d, (730, 235), "total", FT.small, TEXT)
    text(d, (874, 235), "111", FT.mono, CYAN, anchor="ra")

    draw_tensor_badge(d, 795, 274, "[B, T, 111, 2]", CYAN, 1.0)
    text(d, (795, 302), "root-centered [-1, 1]", FT.small, MUTED, anchor="ma")

    breadcrumb(d, 0, p)
    return img


def _step_panel(d, x, w, title, shape, color, glow: float = 0.0):
    box_color = WHITE if glow > 0.45 else STROKE
    box_alpha = 0.40 + 0.60 * glow
    fill_bg = blend(color, 0.08 * glow) if glow > 0.1 else None
    box(d, (x, 78, x + w, 300), box_color, box_alpha, fill=fill_bg)
    text(d, (x + w / 2, 56), title, FT.label, WHITE if glow > 0.3 else TEXT,
         1.0, anchor="ma")
    draw_tensor_badge(d, x + w / 2, 322, shape, color, 1.0, glow)


def scene_pipeline(p: float):
    img, d = new_frame()
    text(d, (W / 2, 26), "Inside the Imitator", FT.title, WHITE, anchor="ma")

    n = 5
    pad, gap = 24, 14
    w = (W - 2 * pad - gap * (n - 1)) / n
    xs = [pad + i * (w + gap) for i in range(n)]
    steps = [
        ("ST-GCN", "[B, 512, T, 111]", CYAN),
        ("node pooling", "[B, T, 512]", SKY),
        ("Transformer · RoPE", "[B, T, 512]", TEAL),
        ("cross-attention", "[B, 20, 512]", INDIGO),
        ("projection", "[B, 20, 3072]", WHITE),
    ]

    # Smooth continuous wave: sweep across stages 0..5 in [0, 0.65], then loop in dwell [0.65, 1.0]
    if p < 0.65:
        wave = (p / 0.65) * 5.0
    else:
        dwell_p = (p - 0.65) / 0.35
        wave = ((dwell_p * 1.5) % 1.0) * 5.0

    # Draw all 5 panels statically from the start (no disjointed pop-in)
    for i, (title, shape, col) in enumerate(steps):
        dist = abs(wave - (i + 0.5))
        glow = max(0.0, 1.0 - dist / 0.75)
        _step_panel(d, xs[i], w, title, shape, col, glow)

        if i > 0:
            x_prev = xs[i] - gap - 1
            x_next = xs[i] - 1
            line(d, [(x_prev, 189), (x_next, 189)], STROKE, 1.5, 0.7)
            line(d, [(x_next - 4, 185), (x_next, 189), (x_next - 4, 193)], STROKE, 1.5, 0.7)
            if abs(wave - i) < 0.6:
                prog = max(0.0, min(1.0, (wave - (i - 0.6)) / 1.2))
                px = x_prev + prog * (x_next - x_prev)
                dot(d, (px, 189), 2.6, WHITE, math.sin(prog * math.pi) * 0.95)

    # 1 ST-GCN: spatial graph pulses in sync with wave
    cx, cy = xs[0] + w / 2, 170
    nodes = [(cx - 42, cy + 44), (cx - 14, cy - 26), (cx + 34, cy + 6),
             (cx + 6, cy + 54), (cx + 44, cy + 58)]
    edges = [(0, 1), (1, 2), (0, 3), (2, 3), (3, 4), (2, 4)]
    st_act = max(0.0, 1.0 - abs(wave - 0.5) / 0.6)
    for u, v in edges:
        line(d, [nodes[u], nodes[v]], CYAN, 1.2, 0.4 + 0.55 * st_act)
    for k, q in enumerate(nodes):
        node_hit = max(0.0, 1.0 - abs(wave - (0.15 + k * 0.14)) / 0.25)
        pulse = 0.2 + 0.8 * node_hit
        dot(d, q, 2.8 + 2.4 * pulse, CYAN, 0.5 + 0.5 * pulse)
    text(d, (cx, cy + 78), "spatial graph", FT.small, MUTED, anchor="ma")
    ty = cy + 96
    for k in range(3):
        t_hit = max(0.0, 1.0 - abs(wave - (0.65 + k * 0.12)) / 0.22)
        line(d, [(cx - 22 + k * 22, ty), (cx - 14 + k * 22, ty)], CYAN, 3.0,
             0.4 + 0.6 * t_hit)
    text(d, (cx, ty + 8), "+ time (k=3)", FT.small, MUTED, anchor="ma")

    # 2 node pooling: 111 nodes collapse into a single feature vector
    cx = xs[1] + w / 2
    rng = np.random.default_rng(7)
    pool_act = max(0.0, 1.0 - abs(wave - 1.5) / 0.45)
    u = min(1.0, max(0.0, (wave - 1.1) / 0.4)) if wave >= 1.0 else 0.0
    for k in range(48):
        ax_ = cx + rng.uniform(-52, 52)
        ay = 150 + rng.uniform(-48, 48)
        bx, by = cx, 150 + (k - 24) * 1.8
        dot(d, (ax_ + (bx - ax_) * u, ay + (by - ay) * u), 1.6, SKY, 0.55 + 0.45 * pool_act)
    text(d, (cx, 240), "mean over 111 nodes", FT.small, WHITE if pool_act > 0.4 else MUTED, anchor="ma")

    # 3 Transformer + RoPE: frame self-attention and position rotation
    cx = xs[2] + w / 2
    for k in range(6):
        block_hit = max(0.0, 1.0 - abs(wave - (2.1 + k * 0.14)) / 0.22)
        x = cx - 63 + k * 22
        d.rounded_rectangle([s(x), s(128), s(x + 14), s(162)], radius=s(3),
                            fill=blend(TEAL, 0.35 + 0.65 * block_hit))
    text(d, (cx, 172), "frame self-attention", FT.small, MUTED, anchor="ma")
    rot_act = max(0.0, 1.0 - abs(wave - 2.5) / 0.5)
    for k in range(3):
        ang = wave * 3.0 + k * 2.1
        ox = cx - 44 + k * 44
        line(d, [(ox, 216), (ox + math.cos(ang) * 14, 216 + math.sin(ang) * 14)],
             TEAL, 1.6 + 0.6 * rot_act, 0.5 + 0.5 * rot_act)
        dot(d, (ox, 216), 1.8, TEAL, 1.0)
    text(d, (cx, 240), "RoPE: rotate by position", FT.small, MUTED, anchor="ma")

    # 4 cross-attention: attention rays sweep into 20 queries
    x0 = xs[3]
    fx = x0 + 26
    qx = x0 + w - 30
    fys = [120 + i_ * 22 for i_ in range(5)]
    qys = [116 + j * 16 for j in range(6)]
    for k, fy in enumerate(fys):
        f_act = max(0.0, 1.0 - abs(wave - (3.1 + k * 0.08)) / 0.2)
        d.rounded_rectangle([s(fx - 6), s(fy - 6), s(fx + 6), s(fy + 6)],
                            radius=s(2), fill=blend(TEAL, 0.4 + 0.6 * f_act))
    q_act = max(0.0, 1.0 - abs(wave - 3.75) / 0.35)
    for j, qy in enumerate(qys):
        dot(d, (qx, qy), 3.2 + 1.0 * q_act, INDIGO, 0.6 + 0.4 * q_act)
    for j, qy in enumerate(qys):
        for k, fy in enumerate(fys):
            sweep_pos = 3.2 + (k * 0.07 + j * 0.05)
            wgt = max(0.0, 1.0 - abs(wave - sweep_pos) / 0.25)
            line(d, [(fx + 8, fy), (qx - 5, qy)], INDIGO, 1.0, 0.20 + 0.80 * wgt)
    text(d, (x0 + w / 2, 240), "20 learned token queries", FT.small, WHITE if q_act > 0.4 else MUTED, anchor="ma")
    text(d, (x0 + w / 2, 256), "Q = queries, K = V = frames", FT.small, MUTED, 0.85, anchor="ma")

    # 5 projection: queries fire beams into Gemma embeddings
    cx = xs[4] + w / 2
    line_act = max(0.0, 1.0 - abs(wave - 4.45) / 0.35)
    out_act = max(0.0, 1.0 - abs(wave - 4.7) / 0.3)
    for j in range(6):
        y = 116 + j * 16
        dot(d, (cx - 46, y), 3.0, INDIGO, 1.0)
        line(d, [(cx - 40, y), (cx + 40, y)], INDIGO, 1.0 + 0.8 * line_act,
             0.30 + 0.70 * line_act)
        dot(d, (cx + 46, y), 3.0 + 1.5 * out_act, WHITE, 0.5 + 0.5 * out_act)
    text(d, (cx, 230), "Linear 512 → 3072", FT.small, WHITE if line_act > 0.4 else MUTED, anchor="ma")
    text(d, (cx, 248), "Gemma embedding space", FT.small, WHITE if out_act > 0.4 else MUTED, anchor="ma")

    breadcrumb(d, 1, p)
    return img


def scene_output(p: float):
    img, d = new_frame()
    text(d, (W / 2, 26), "From token IDs to text", FT.title, WHITE, anchor="ma")

    # Reveal completes by p = 0.35; remaining 65% (4.4 seconds) is peaceful dwell time
    card_w, card_h = 142, 86
    gap = 16
    total_w = 4 * card_w + 3 * gap
    x0 = (W - total_w) / 2
    y0 = 62

    for i, (tid, sub) in enumerate(zip(TOKEN_IDS, SUBTOKENS)):
        a = min(1.0, max(0.0, (p - 0.04 * i) / 0.10))
        x = x0 + i * (card_w + gap)
        # Card container with high contrast and subtle glow
        fill_col = blend((18, 26, 44), a)
        stroke_col = blend(INDIGO, a * 0.85)
        d.rounded_rectangle([s(x), s(y0), s(x + card_w), s(y0 + card_h)],
                            radius=s(8), fill=fill_col, outline=stroke_col,
                            width=max(1, int(1.5 * SS)))
        text(d, (x + card_w / 2, y0 + 15), f"TOKEN {i+1}", FT.badge, SKY, a, anchor="mm")
        text(d, (x + card_w / 2, y0 + 40), str(tid), FT.mono_card_id, CYAN, a, anchor="mm")
        text(d, (x + card_w / 2, y0 + 66), f"“{sub}”", FT.mono_card_sub, WHITE, a, anchor="mm")

    # Detokenizer flow pill badge & chevron
    flow_a = min(1.0, max(0.0, (p - 0.14) / 0.10))
    flow_txt = "tokenizer.decode( [236754, 1525, 604, 14246] )  →  detokenize & assemble"
    fw = FT.mono_small.getlength(flow_txt) / SS
    pw, ph = fw + 24, 26
    px0, py0 = (W - pw) / 2, 160
    if flow_a > 0.01:
        d.rounded_rectangle([s(px0), s(py0), s(px0 + pw), s(py0 + ph)],
                            radius=s(13), fill=blend((16, 24, 38), flow_a),
                            outline=blend(STROKE, flow_a * 0.9), width=max(1, int(1.2 * SS)))
    text(d, (W / 2, 173), flow_txt, FT.mono_small, TEXT, flow_a, anchor="mm")

    # Chevron pointing smoothly into hero card
    line(d, [(W / 2 - 7, 193), (W / 2, 199), (W / 2 + 7, 193)], TEAL, 1.8, flow_a)

    # Center Hero Card: Massive, radiant, ultra-legible
    hero_a = min(1.0, max(0.0, (p - 0.20) / 0.12))
    hero_w, hero_h = 660, 122
    hx0, hy0 = (W - hero_w) / 2, 206

    card_fill = blend((16, 28, 46), hero_a)
    card_border = blend(TEAL, hero_a * 0.90)
    d.rounded_rectangle([s(hx0), s(hy0), s(hx0 + hero_w), s(hy0 + hero_h)],
                        radius=s(10), fill=card_fill, outline=card_border,
                        width=max(1, int(2.0 * SS)))

    # Status pill
    draw_pill(d, W / 2, hy0 + 21, "DECODED TRANSCRIPTION · EXACT MATCH ✓", TEAL, hero_a)

    # Hero text: 34pt bold white
    text(d, (W / 2, hy0 + 64), f"“{PRED_TEXT}”", FT.hero, WHITE, hero_a, anchor="mm")

    # Details line
    text(d, (W / 2, hy0 + 102), f"reference gloss: “{GLOSS}”   ·   clip 1660   ·   teacher-forced checkpoint output",
         FT.mono_small, TEXT, hero_a * 0.95, anchor="mm")

    # Subtle footnote at bottom
    foot_a = min(1.0, max(0.0, (p - 0.26) / 0.10))
    text(d, (W / 2, 352), "Optional 2nd stage (frozen Gemma) adds casing & punctuation · inference completes in 4 tokens",
         FT.small, MUTED, foot_a * 0.85, anchor="ma")

    breadcrumb(d, 2, p)
    return img


# --- assembly ---------------------------------------------------------------

TIMELINE = [(scene_capture, 4.0), (scene_pipeline, 9.5), (scene_output, 6.8)]
FADE = 3  # frames faded in/out at each scene boundary


def render_frames(fps: int):
    bg = Image.new("RGB", (W, H), BG)
    frames = []
    for fn, dur in TIMELINE:
        n = max(2, int(round(dur * fps)))
        scene = []
        for i in range(n):
            img = fn(i / (n - 1)).resize((W, H), Image.LANCZOS)
            scene.append(img)
        for i in range(min(FADE, n // 2)):
            k = (i + 1) / (FADE + 1)
            scene[i] = Image.blend(bg, scene[i], k)
            scene[-1 - i] = Image.blend(bg, scene[-1 - i], k)
        frames.extend(scene)
    return frames


def write_gif(frames, out: Path, fps: int, colors: int):
    # One shared palette for every frame, so Pillow can emit inter-frame deltas
    # instead of a full image per frame (that alone is most of the file size).
    master = frames[len(frames) // 2].convert(
        "P", palette=Image.ADAPTIVE, colors=colors)
    pal = [f.quantize(palette=master, dither=Image.NONE) for f in frames]
    out.parent.mkdir(parents=True, exist_ok=True)
    pal[0].save(out, save_all=True, append_images=pal[1:],
                duration=int(1000 / fps), loop=0, optimize=True)


def selftest():
    assert clip()["kp"].shape[1:] == (111, 2), clip()["kp"].shape
    for t in (0.0, 0.3, 1.0):
        pts = frame_points(t, 200, 200, 300)
        assert pts.shape == (111, 2) and np.isfinite(pts).all(), f"bad frame t={t}"
        assert len(bone_segments(pts)) == len(POSE_EDGES) + 2 + 2 * len(HAND_EDGES)
        assert 0 < len(bone_segments(pts, max_len=0.22 * 300)) < len(bone_segments(pts))
    for fn in (scene_capture, scene_pipeline, scene_output):
        img = fn(0.6).resize((W, H), Image.LANCZOS)
        assert img.size == (W, H)
        assert len(img.getcolors(maxcolors=1 << 20)) > 8, f"{fn.__name__} is blank"
    frames = render_frames(2)
    assert len(frames) == sum(max(2, int(round(d * 2))) for _, d in TIMELINE)
    print("selftest ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parents[2] / "docs" / "imitator.gif")
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--colors", type=int, default=64)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    global FT
    FT = Fonts()

    if args.selftest:
        selftest()
        return

    frames = render_frames(args.fps)
    write_gif(frames, args.out, args.fps, args.colors)
    mb = args.out.stat().st_size / 1e6
    print(f"{args.out}: {len(frames)} frames, {W}x{H}, {mb:.2f} MB")
    if mb > 2.0:
        print("warning: over 2 MB — try --colors 48 or --fps 10")


if __name__ == "__main__":
    main()
