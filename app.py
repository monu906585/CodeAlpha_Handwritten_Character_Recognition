import hashlib
import io
import os

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

st.set_page_config(page_title="Multi-Digit Recognition", page_icon="✍️", layout="wide")

st.title("✍️ Multi-Digit Handwritten Recognition")
st.write("Upload or capture a photo containing handwritten digits.")

MODEL_PATH = "models/handwritten_digit_model.keras"
MAX_SIDE = 1600  # photos are resized to this so kernel sizes behave consistently


# ----------------------------------------------------------------------
# Sidebar settings
# ----------------------------------------------------------------------
st.sidebar.header("Settings")

with st.sidebar.expander("Image", expanded=True):
    invert_input = st.checkbox(
        "Light digits on dark background", value=False,
        help="Tick this for chalkboard-style images.",
    )
    method = st.selectbox(
        "Thresholding method",
        ["Otsu (clean paper)", "Adaptive (shadows / uneven light)"],
        help="Try Adaptive if the page has strong shadows or the digits break apart.",
    )
    remove_lines = st.checkbox("Remove ruled / notebook lines", value=True)
    remove_vlines = st.checkbox(
        "Also remove vertical grid lines", value=False,
        help="For graph paper. May erase a very tall, thin '1'.",
    )

with st.sidebar.expander("Detection", expanded=True):
    min_height_pct = st.slider(
        "Minimum digit height (% of image height)", 0.5, 10.0, 1.5, 0.5
    )
    split_touching = st.checkbox("Try to split touching digits", value=True)
    gap_factor = st.slider(
        "Gap that starts a new number (× digit height, 0 = off)", 0.0, 3.0, 1.0, 0.1,
        help="Large gaps between digits in a row are shown as spaces.",
    )

with st.sidebar.expander("Recognition", expanded=True):
    auto_stroke = st.checkbox(
        "Auto-normalise stroke thickness", value=True,
        help="Makes thin pen strokes look like the thick strokes the model was trained on.",
    )
    thicken = st.slider(
        "Manual thickening", 0, 3, 1, disabled=auto_stroke,
        help="Only used when auto-normalise is off.",
    )
    use_tta = st.checkbox("Test-time augmentation (slower, more stable)", value=True)
    low_conf = st.slider("Flag predictions below confidence", 0.50, 0.99, 0.80, 0.01)


@st.cache_resource
def load_model():
    import tensorflow as tf
    return tf.keras.models.load_model(MODEL_PATH)


# ----------------------------------------------------------------------
# Preprocessing
# ----------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_image(data: bytes):
    """Open image, fix phone-camera rotation, and limit its size."""
    image = Image.open(io.BytesIO(data))
    image = ImageOps.exif_transpose(image).convert("RGB")
    rgb = np.array(image)

    h, w = rgb.shape[:2]
    scale = MAX_SIDE / max(h, w)
    if scale < 1:
        rgb = cv2.resize(rgb, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_AREA)
    return rgb


def binarize(rgb, invert=False, adaptive=False, remove_hlines=True, remove_vlines=False):
    """Return a clean binary image: white strokes on black background."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if invert:
        gray = 255 - gray

    # Estimate paper background (closing removes the dark pen strokes),
    # then divide it out to flatten shadows.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    background = cv2.GaussianBlur(background, (0, 0), 15)
    flat = cv2.divide(gray, background, scale=255)
    flat = cv2.GaussianBlur(flat, (3, 3), 0)

    if adaptive:
        block = max(15, (min(flat.shape) // 20) | 1)
        binary = cv2.adaptiveThreshold(
            flat, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, block, 10,
        )
    else:
        _, binary = cv2.threshold(
            flat, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

    h, w = binary.shape
    if remove_hlines:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (max(40, w // 10), 1))
        lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k)
        lines = cv2.dilate(lines, np.ones((3, 1), np.uint8))
        binary = cv2.subtract(binary, lines)

    if remove_vlines:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(40, h // 4)))
        lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k)
        lines = cv2.dilate(lines, np.ones((1, 3), np.uint8))
        binary = cv2.subtract(binary, lines)

    # Remove tiny specks.
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))


# ----------------------------------------------------------------------
# Digit detection
# ----------------------------------------------------------------------
def union(a, b):
    x1 = min(a[0], b[0])
    y1 = min(a[1], b[1])
    x2 = max(a[0] + a[2], b[0] + b[2])
    y2 = max(a[1] + a[3], b[1] + b[3])
    return (x1, y1, x2 - x1, y2 - y1)


def should_merge(a, b, typical_h):
    """Merge pieces of one digit (e.g. a detached top bar of a 5 or 7)."""
    overlap = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
    if overlap < 0.5 * min(a[2], b[2]):
        return False
    gap = max(a[1], b[1]) - min(a[1] + a[3], b[1] + b[3])
    if gap > 0.25 * typical_h:
        return False
    return union(a, b)[3] <= 1.6 * typical_h


def merge_boxes(boxes, typical_h):
    boxes = list(boxes)
    changed = True
    while changed:
        changed = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                if should_merge(boxes[i], boxes[j], typical_h):
                    boxes[i] = union(boxes[i], boxes[j])
                    del boxes[j]
                    changed = True
                    break
            if changed:
                break
    return boxes


def tighten(binary, box):
    """Shrink a box to the actual stroke pixels inside it."""
    x, y, w, h = box
    ys, xs = np.where(binary[y:y + h, x:x + w] > 0)
    if len(xs) == 0:
        return None
    return (x + int(xs.min()), y + int(ys.min()),
            int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))


def split_wide_box(binary, box):
    """Split a wide box (likely touching digits) at the thinnest columns."""
    x, y, w, h = box
    if w / h < 1.3:
        return [box]

    n = max(2, int(round(w / (0.65 * h))))
    profile = (binary[y:y + h, x:x + w] > 0).sum(axis=0).astype(float)
    profile = np.convolve(profile, np.ones(3) / 3, mode="same")

    cuts = [0]
    for i in range(1, n):
        expected = int(w * i / n)
        margin = int(w / n * 0.3)
        lo = max(cuts[-1] + 3, expected - margin)
        hi = min(w - 3, expected + margin)
        if hi > lo:
            cuts.append(lo + int(np.argmin(profile[lo:hi])))
    cuts.append(w)

    parts = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        if b - a < 3:
            continue
        t = tighten(binary, (x + a, y, b - a, h))
        if t:
            parts.append(t)
    return parts or [box]


def reading_order(boxes):
    """Group boxes into text rows, then sort each row left to right."""
    if not boxes:
        return []

    typical_h = float(np.median([b[3] for b in boxes]))
    boxes = sorted(boxes, key=lambda b: b[1] + b[3] / 2)

    rows = [[boxes[0]]]
    for b in boxes[1:]:
        cy = b[1] + b[3] / 2
        row_cy = np.mean([r[1] + r[3] / 2 for r in rows[-1]])
        if abs(cy - row_cy) > 0.6 * typical_h:
            rows.append([b])
        else:
            rows[-1].append(b)

    return [sorted(row, key=lambda b: b[0]) for row in rows]


def group_row(row, factor):
    """Split a row into separate numbers wherever there is a large gap."""
    if factor <= 0 or len(row) < 2:
        return [row]
    typical_h = float(np.median([b[3] for b in row]))
    groups = [[row[0]]]
    for prev, b in zip(row, row[1:]):
        gap = b[0] - (prev[0] + prev[2])
        if gap > factor * typical_h:
            groups.append([b])
        else:
            groups[-1].append(b)
    return groups


def detect_digits(binary, min_height_pct, split_touching):
    img_h, img_w = binary.shape

    # Slight dilation joins broken strokes into one component.
    joined = cv2.dilate(binary, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(joined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_size = img_h * min_height_pct / 100.0
    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if max(w, h) < min_size:
            continue
        if w > img_w * 0.6 or h > img_h * 0.6:
            continue
        if w > 4 * h or h > 12 * w:  # leftover lines, not digits
            continue
        boxes.append((x, y, w, h))

    if not boxes:
        return []

    typical_h = float(np.percentile([b[3] for b in boxes], 70))
    boxes = merge_boxes(boxes, typical_h)

    # Tighten to real stroke pixels and drop small leftovers (dots, smudges).
    boxes = [t for t in (tighten(binary, b) for b in boxes) if t]
    boxes = [b for b in boxes if b[3] >= 0.3 * typical_h]

    if split_touching:
        split = []
        for b in boxes:
            split.extend(split_wide_box(binary, b))
        boxes = split

    return reading_order(boxes)


# ----------------------------------------------------------------------
# MNIST-style preparation
# ----------------------------------------------------------------------
def stroke_width(crop):
    """Rough stroke width in pixels using the distance transform."""
    dt = cv2.distanceTransform(crop, cv2.DIST_L2, 3)
    vals = dt[dt > 0]
    return 2.0 * float(np.percentile(vals, 95)) if vals.size else 0.0


def prepare_digit(binary, box, thicken=1, auto_stroke=True):
    """
    One detected digit -> 28x28 MNIST-style input:
    tight crop -> stroke normalisation -> fit in 20x20 ->
    place on 28x28 -> centre of mass at (14, 14).
    """
    x, y, w, h = box
    crop = binary[y:y + h, x:x + w].copy()
    side = max(h, w)

    if auto_stroke:
        target = 0.14 * side
        k = int(round(target - stroke_width(crop))) + 1
        k = min(k, max(2, int(0.15 * side)))
    else:
        k = int(round(side * 0.05 * thicken)) if thicken > 0 else 0
    if k >= 2:
        crop = cv2.dilate(crop, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))

    scale = 20.0 / max(crop.shape)
    new_w = max(1, int(round(crop.shape[1] * scale)))
    new_h = max(1, int(round(crop.shape[0] * scale)))
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((28, 28), dtype=np.uint8)
    ox, oy = (28 - new_w) // 2, (28 - new_h) // 2
    canvas[oy:oy + new_h, ox:ox + new_w] = resized

    m = cv2.moments(canvas)
    if m["m00"] > 0:
        cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
        shift = np.float32([[1, 0, 14 - cx], [0, 1, 14 - cy]])
        canvas = cv2.warpAffine(canvas, shift, (28, 28))

    return canvas.astype("float32") / 255.0


def predict_digits(model, images, use_tta):
    """images: (N, 28, 28) float32. Returns (N, 10) probabilities."""
    n = len(images)
    shifts = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)] if use_tta else [(0, 0)]

    # One predict() call for every shifted copy is much faster than a loop.
    variants = [np.roll(images, shift=(dy, dx), axis=(1, 2)) for dx, dy in shifts]
    batch = np.concatenate(variants, axis=0)

    in_shape = model.input_shape
    if len(in_shape) == 2:  # flat dense model
        batch = batch.reshape(len(batch), -1)
    else:
        batch = batch.reshape(-1, 28, 28, 1)

    probs = model.predict(batch, verbose=0)
    return probs.reshape(len(shifts), n, -1).mean(axis=0)


# ----------------------------------------------------------------------
# App
# ----------------------------------------------------------------------
if not os.path.exists(MODEL_PATH):
    st.error("Model not found. Run train_model.py first.")
    st.stop()

source = st.radio("Image source", ["Upload", "Camera"], horizontal=True)
if source == "Upload":
    uploaded_file = st.file_uploader(
        "Upload a page containing handwritten digits", type=["png", "jpg", "jpeg"]
    )
else:
    uploaded_file = st.camera_input("Take a photo of the digits")

if not uploaded_file:
    st.info("Tips: good light, dark pen on white paper, and a photo taken straight on.")
    st.stop()

rgb = load_image(uploaded_file.getvalue())

with st.spinner("Detecting digits..."):
    binary = binarize(
        rgb,
        invert=invert_input,
        adaptive=method.startswith("Adaptive"),
        remove_hlines=remove_lines,
        remove_vlines=remove_vlines,
    )
    rows = detect_digits(binary, min_height_pct, split_touching)

boxes = [b for row in rows for b in row]

if not boxes:
    st.image(binary, caption="Threshold image", width=400)
    st.warning(
        "No digits detected. Try a clearer image, lower the minimum digit height, "
        "switch the thresholding method, or tick the dark-background option."
    )
    st.stop()

try:
    model = load_model()
except Exception as e:  # noqa: BLE001
    st.error(f"Could not load the model: {e}")
    st.stop()

with st.spinner("Recognising..."):
    prepared = np.stack([prepare_digit(binary, b, thicken, auto_stroke) for b in boxes])
    probs = predict_digits(model, prepared, use_tta)

top = probs.argmax(axis=1)
conf = probs.max(axis=1)
second = np.argsort(probs, axis=1)[:, -2]
n_low = int((conf < low_conf).sum())

# ---- Summary ---------------------------------------------------------
m1, m2, m3, m4 = st.columns(4)
m1.metric("Digits found", len(boxes))
m2.metric("Rows", len(rows))
m3.metric("Low confidence", n_low)
m4.metric("Average confidence", f"{conf.mean():.1%}")

# ---- Annotated image -------------------------------------------------
marked = rgb.copy()
font_scale = max(0.6, marked.shape[0] / 1000)
thick = max(2, int(font_scale * 2))
for i, (x, y, w, h) in enumerate(boxes):
    color = (0, 200, 0) if conf[i] >= low_conf else (255, 0, 0)
    cv2.rectangle(marked, (x, y), (x + w, y + h), color, 2)
    cv2.putText(marked, str(top[i]), (x, max(20, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thick)

ok, png = cv2.imencode(".png", cv2.cvtColor(marked, cv2.COLOR_RGB2BGR))

# ---- Results table (editable) ---------------------------------------
records = []
i = 0
for r, row in enumerate(rows, start=1):
    for g, group in enumerate(group_row(row, gap_factor), start=1):
        for _ in group:
            records.append({
                "#": i + 1,
                "Row": r,
                "Number": g,
                "Predicted": str(int(top[i])),
                "Confidence": float(conf[i]),
                "Second guess": f"{int(second[i])} ({probs[i][second[i]]:.0%})",
                "Digit": str(int(top[i])),
            })
            i += 1
df = pd.DataFrame(records)

signature = hashlib.md5(
    (str(boxes) + str(top.tolist()) + str(gap_factor)).encode()
).hexdigest()[:10]

tab_results, tab_crops, tab_debug = st.tabs(["Results", "Digit crops", "Preprocessing"])

with tab_results:
    left, right = st.columns([3, 2])

    with left:
        st.subheader("Detected digits")
        st.image(marked, width=700,
                 caption="Green = confident, red = low confidence")
        if ok:
            st.download_button("Download annotated image", png.tobytes(),
                               "annotated.png", "image/png")

    with right:
        st.subheader("Check and correct")
        st.caption("Fix any wrong digit in the last column; the sequence updates below.")
        edited = st.data_editor(
            df,
            key=f"editor_{signature}",
            hide_index=True,
            disabled=["#", "Row", "Number", "Predicted", "Confidence", "Second guess"],
            column_config={
                "Confidence": st.column_config.ProgressColumn(
                    "Confidence", min_value=0.0, max_value=1.0, format="%.0f%%"
                ),
                "Digit": st.column_config.SelectboxColumn(
                    "Digit (edit)", options=[str(d) for d in range(10)], required=True
                ),
            },
        )

        st.subheader("Detected sequence")
        lines = []
        for r, rdf in edited.groupby("Row", sort=True):
            parts = ["".join(g["Digit"]) for _, g in rdf.groupby("Number", sort=True)]
            line = " ".join(parts)
            lines.append(line)
            st.success(f"Row {r}: {line}")

        c1, c2 = st.columns(2)
        c1.download_button("Download CSV", edited.to_csv(index=False),
                        "digits.csv", "text/csv")
        c2.download_button("Download text", "\n".join(lines), "digits.txt", "text/plain")

with tab_crops:
    only_low = st.checkbox("Show only low-confidence digits", value=False)
    indices = [k for k in range(len(boxes)) if (not only_low or conf[k] < low_conf)]
    if not indices:
        st.info("Nothing below the confidence threshold.")
    per_row = 6
    for start in range(0, len(indices), per_row):
        cols = st.columns(per_row)
        for col, k in zip(cols, indices[start:start + per_row]):
            with col, st.container(border=True):
                thumb = cv2.resize((prepared[k] * 255).astype(np.uint8), (112, 112),
                                interpolation=cv2.INTER_NEAREST)
                st.image(thumb, caption=f"#{k + 1} (model input)")
                st.markdown(f"**Predicted: {top[k]}**")
                st.caption(f"{conf[k]:.1%} · 2nd: {second[k]} ({probs[k][second[k]]:.0%})")
                if conf[k] < low_conf:
                    st.warning("Check this one", icon="⚠️")

with tab_debug:
    d1, d2 = st.columns(2)
    d1.image(rgb, width=450, caption="Original (resized)")
    d2.image(binary, width=450, caption="Threshold: white strokes on black")

st.caption(
    "Touching digits are split with a heuristic and may occasionally be wrong. "
    "Check red (low confidence) boxes manually."
)