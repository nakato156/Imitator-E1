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
BG = (13, 17, 23)

BLUE = (108, 173, 255)    # keypoints / bones
DIM_BLUE = (60, 105, 160)
ORANGE = (232, 163, 76)   # ST-GCN graph
GREEN = (95, 211, 155)    # frame sequence
PURPLE = (197, 138, 249)  # token queries
TEXT = (201, 212, 227)
MUTED = (138, 151, 168)
WHITE = (242, 245, 249)
STROKE = (124, 138, 160)

# Real prediction for the same clip the keypoints come from: 1660, gloss
# "Hambriento", predicted == target. Picked over the other exact-match rows
# because both hands are cleanly detected in its keypoints.
# outputs/v126_temporal/diag_imitator_A2_predlen_optunaStable_20260625_194336/predictions.jsonl
TOKEN_IDS = [236754, 1525, 604, 14246]
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
        self.mono = font(12, mono=True)
        self.mono_big = font(20, mono=True)
        self.big = font(24, bold=True)


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


def scene_capture(p: float):
    img, d = new_frame()
    c = clip()
    text(d, (W / 2, 26), "From video to keypoints", FT.title, WHITE, anchor="ma")

    # left: the clip itself, as a skeleton
    box(d, (70, 60, 360, 350))
    text(d, (78, 44), f"clip {c['clip_id']} · “{c['gloss']}”",
         FT.small, MUTED)
    pts = frame_points(p, 218, 205, 250)

    fade = min(1.0, max(0.0, (p - 0.35) / 0.3))   # bones dissolve, dots remain
    for b in bone_segments(pts):
        line(d, b, BLUE, 1.6, 0.85 * (1 - 0.75 * fade))
    for q in pts:
        dot(d, q, 1.4 + 0.8 * fade, BLUE, 0.35 + 0.6 * fade)

    line(d, [(378, 205), (438, 205)], STROKE, 1.5)
    line(d, [(430, 200), (438, 205), (430, 210)], STROKE, 1.5)

    # right: the same frame, coloured by the region each keypoint belongs to
    box(d, (455, 60, 890, 350))
    text(d, (463, 44), "pose estimation", FT.small, MUTED)
    pts2 = frame_points(p, 598, 205, 250)
    for sl, col in REGIONS:
        for q in pts2[sl]:
            dot(d, q, 1.7, col, 0.9)

    rows = [
        ("pose", 7, BLUE),
        ("face", 64, DIM_BLUE),
        ("left hand", 20, GREEN),
        ("right hand", 20, PURPLE),
    ]
    y = 128
    for i, (name, n, c) in enumerate(rows):
        a = min(1.0, max(0.0, (p - 0.15 * i) / 0.18))
        dot(d, (742, y - 4), 3.5, c, a)
        text(d, (756, y - 11), f"{name}", FT.small, TEXT, a)
        text(d, (872, y - 11), f"{n}", FT.mono, c, a, anchor="ra")
        y += 30
    line(d, [(742, 238), (872, 238)], STROKE, 1.2, 0.5)
    text(d, (756, 245), "total", FT.small, TEXT)
    text(d, (872, 245), "111", FT.mono, WHITE, anchor="ra")
    text(d, (742, 282), "[B, T, 111, 2]", FT.mono, WHITE)
    text(d, (742, 302), "T frames  ·  (x, y)", FT.small, MUTED)

    breadcrumb(d, 0, p)
    return img


def _step_panel(d, x, w, title, shape, alpha, active):
    box(d, (x, 78, x + w, 300), STROKE, 0.3 + 0.7 * alpha)
    text(d, (x + w / 2, 56), title, FT.label, WHITE if active else TEXT,
         alpha, anchor="ma")
    text(d, (x + w / 2, 308), shape, FT.mono, MUTED, alpha * 0.9, anchor="ma")


def scene_pipeline(p: float):
    img, d = new_frame()
    text(d, (W / 2, 26), "Inside the Imitator", FT.title, WHITE, anchor="ma")

    n = 5
    pad, gap = 24, 14
    w = (W - 2 * pad - gap * (n - 1)) / n
    xs = [pad + i * (w + gap) for i in range(n)]
    steps = [
        ("ST-GCN", "[B, 512, T, 111]"),
        ("node pooling", "[B, T, 512]"),
        ("Transformer · RoPE", "[B, T, 512]"),
        ("cross-attention", "[B, 20, 512]"),
        ("projection", "[B, 20, 3072]"),
    ]
    # reveal one step per slice of p, everything stays visible afterwards
    for i, (title, shape) in enumerate(steps):
        a = min(1.0, max(0.0, (p - i * 0.16) / 0.12))
        active = abs(p - (i * 0.16 + 0.10)) < 0.11
        _step_panel(d, xs[i], w, title, shape, a, active)
        if i:
            la = min(1.0, max(0.0, (p - i * 0.16 + 0.04) / 0.08))
            line(d, [(xs[i] - gap - 1, 189), (xs[i] - 1, 189)], STROKE, 1.5, la)

    beat = p * 5.0

    # 1 ST-GCN: skeleton patch, pulses hopping along edges
    a = min(1.0, max(0.0, p / 0.12))
    cx, cy = xs[0] + w / 2, 170
    nodes = [(cx - 42, cy + 44), (cx - 14, cy - 26), (cx + 34, cy + 6),
             (cx + 6, cy + 54), (cx + 44, cy + 58)]
    edges = [(0, 1), (1, 2), (0, 3), (2, 3), (3, 4), (2, 4)]
    for u, v in edges:
        line(d, [nodes[u], nodes[v]], ORANGE, 1.2, a * 0.45)
    for i, q in enumerate(nodes):
        pulse = 0.5 + 0.5 * math.sin(beat * 2.2 - i * 1.1)
        dot(d, q, 3.0 + 2.5 * pulse, ORANGE, a * (0.45 + 0.55 * pulse))
    text(d, (cx, cy + 78), "spatial graph", FT.small, MUTED, a, anchor="ma")
    ty = cy + 96
    for k in range(3):
        line(d, [(cx - 22 + k * 22, ty), (cx - 14 + k * 22, ty)], ORANGE, 3.0,
             a * (0.35 + 0.65 * (0.5 + 0.5 * math.sin(beat * 2.2 - k * 0.9))))
    text(d, (cx, ty + 8), "+ time (k=3)", FT.small, MUTED, a, anchor="ma")

    # 2 node pooling: 111 node dots collapse into one column
    a = min(1.0, max(0.0, (p - 0.16) / 0.12))
    cx = xs[1] + w / 2
    u = min(1.0, max(0.0, (p - 0.18) / 0.14))
    rng = np.random.default_rng(7)
    for i in range(48):
        ax_ = cx + rng.uniform(-52, 52)
        ay = 150 + rng.uniform(-48, 48)
        bx, by = cx, 150 + (i - 24) * 1.8
        dot(d, (ax_ + (bx - ax_) * u, ay + (by - ay) * u), 1.6, BLUE, a * 0.85)
    text(d, (cx, 240), "mean over 111 nodes", FT.small, MUTED, a, anchor="ma")

    # 3 Transformer + RoPE
    a = min(1.0, max(0.0, (p - 0.32) / 0.12))
    cx = xs[2] + w / 2
    for i in range(6):
        pulse = 0.35 + 0.65 * (0.5 + 0.5 * math.sin(beat * 2.0 - i * 0.7))
        x = cx - 63 + i * 22
        d.rounded_rectangle([s(x), s(128), s(x + 14), s(162)], radius=s(3),
                            fill=blend(GREEN, a * pulse))
    text(d, (cx, 172), "frame self-attention", FT.small, MUTED, a, anchor="ma")
    for i in range(3):
        ang = beat * 1.1 + i * 2.1
        ox = cx - 44 + i * 44
        line(d, [(ox, 216), (ox + math.cos(ang) * 14, 216 + math.sin(ang) * 14)],
             GREEN, 1.6, a)
        dot(d, (ox, 216), 1.8, GREEN, a)
    text(d, (cx, 240), "RoPE: rotate by position", FT.small, MUTED, a, anchor="ma")

    # 4 cross-attention: 20 learned queries read the frames
    a = min(1.0, max(0.0, (p - 0.48) / 0.12))
    x0 = xs[3]
    fx = x0 + 26
    qx = x0 + w - 30
    fys = [120 + i * 22 for i in range(5)]
    qys = [116 + i * 16 for i in range(6)]
    for i, fy in enumerate(fys):
        d.rounded_rectangle([s(fx - 6), s(fy - 6), s(fx + 6), s(fy + 6)],
                            radius=s(2), fill=blend(GREEN, a * 0.8))
    for j, qy in enumerate(qys):
        dot(d, (qx, qy), 3.2, PURPLE, a)
    for j, qy in enumerate(qys):
        for i, fy in enumerate(fys):
            wgt = math.exp(-((i - (2.0 + 1.8 * math.sin(beat * 1.3 + j * 0.9))) ** 2) / 1.6)
            line(d, [(fx + 8, fy), (qx - 5, qy)], PURPLE, 1.0, a * wgt * 0.85)
    text(d, (x0 + w / 2, 240), "20 learned token queries", FT.small, MUTED, a,
         anchor="ma")
    text(d, (x0 + w / 2, 256), "Q = queries, K = V = frames", FT.small, MUTED,
         a * 0.8, anchor="ma")

    # 5 projection into Gemma embedding space
    a = min(1.0, max(0.0, (p - 0.64) / 0.12))
    cx = xs[4] + w / 2
    for j in range(6):
        y = 116 + j * 16
        dot(d, (cx - 46, y), 3.0, PURPLE, a)
        line(d, [(cx - 40, y), (cx + 40, y)], PURPLE, 1.0,
             a * (0.3 + 0.5 * (0.5 + 0.5 * math.sin(beat * 2.0 - j * 0.8))))
        dot(d, (cx + 46, y), 3.0, WHITE, a * 0.9)
    text(d, (cx, 230), "Linear 512 → 3072", FT.small, MUTED, a, anchor="ma")
    text(d, (cx, 248), "Gemma embedding space", FT.small, MUTED, a, anchor="ma")

    breadcrumb(d, 1, p)
    return img


def scene_output(p: float):
    img, d = new_frame()
    text(d, (W / 2, 26), "From token IDs to text", FT.title, WHITE, anchor="ma")

    box(d, (70, 74, 420, 240))
    text(d, (78, 58), "argmax over the Gemma vocabulary", FT.small, MUTED)
    for i, tid in enumerate(TOKEN_IDS):
        a = min(1.0, max(0.0, (p - 0.06 * i) / 0.12))
        x = 96 + i * 82
        d.rounded_rectangle([s(x), s(120), s(x + 66), s(154)], radius=s(5),
                            outline=blend(PURPLE, a), width=int(1.5 * SS))
        text(d, (x + 33, 128), str(tid), FT.mono, WHITE, a, anchor="ma")
    text(d, (245, 186), "4 token IDs  ·  [B, 20, vocab] → argmax", FT.small,
         MUTED, min(1.0, p / 0.3), anchor="ma")
    text(d, (245, 206), "tokenizer.decode(skip_special_tokens=True)", FT.small,
         MUTED, min(1.0, max(0.0, (p - 0.3) / 0.2)), anchor="ma")

    a = min(1.0, max(0.0, (p - 0.38) / 0.12))
    line(d, [(432, 157), (492, 157)], STROKE, 1.5, a)
    line(d, [(484, 152), (492, 157), (484, 162)], STROKE, 1.5, a)

    a = min(1.0, max(0.0, (p - 0.44) / 0.16))
    box(d, (508, 74, 890, 240), STROKE, a)
    text(d, (516, 58), "decoded transcription", FT.small, MUTED, a)
    text(d, (699, 118), PRED_TEXT, FT.big, WHITE, a, anchor="ma")
    text(d, (699, 168), f"reference gloss: {GLOSS}", FT.small, MUTED, a,
         anchor="ma")
    text(d, (699, 192), "real checkpoint output · teacher-forced length",
         FT.small, GREEN, a * 0.9, anchor="ma")

    a = min(1.0, max(0.0, (p - 0.68) / 0.18))
    box(d, (240, 272, 720, 348), STROKE, a * 0.6)
    text(d, (480, 288), "optional second stage", FT.small, MUTED, a * 0.8,
         anchor="ma")
    text(d, (480, 308), "frozen Gemma · punctuation and formatting", FT.small,
         TEXT, a, anchor="ma")
    text(d, (480, 326), "run separately, not part of the Imitator forward pass",
         FT.small, MUTED, a * 0.7, anchor="ma")

    breadcrumb(d, 2, p)
    return img


# --- assembly ---------------------------------------------------------------

TIMELINE = [(scene_capture, 3.0), (scene_pipeline, 7.5), (scene_output, 3.5)]
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
