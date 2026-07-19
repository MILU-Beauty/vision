from maix import camera, image, display, time, touchscreen
import math

# Workflow.
# Use "a4_tape" for the black-tape rectangle task.
WORKFLOW = "a4_tape"  # "screen_reset" or "a4_tape"

# Camera profiles.
# Switch this line first when the venue light changes.
if WORKFLOW == "a4_tape":
    CAMERA_PROFILE = "a4_dark"
else:
    CAMERA_PROFILE = "laser_dark"

CAMERA_PROFILES = {
    "laser_dark": {"exposure": 2600, "gain": 65},
    "laser_normal": {"exposure": 1300, "gain": 65},
    "laser_bright": {"exposure": 700, "gain": 50},
    "a4_dark": {"exposure": 6500, "gain": 80},
    "a4_normal": {"exposure": 3600, "gain": 70},
    "a4_bright": {"exposure": 1800, "gain": 55},
    "paper": {"exposure": 1200, "gain": 80},
}

# Screen model.
IMG_W = 320
IMG_H = 240
SCREEN_SIDE_CM = 50.0
SCREEN_HALF_CM = SCREEN_SIDE_CM / 2.0
CM_PER_PIXEL_FALLBACK = SCREEN_SIDE_CM / float(IMG_W)

ORIGIN_X = IMG_W // 2
ORIGIN_Y = IMG_H // 2
RESET_ERROR_LIMIT = 2.0
RESET_ERROR_LIMIT_PX = max(1, int(RESET_ERROR_LIMIT / CM_PER_PIXEL_FALLBACK + 0.5))

# Red laser detection.
# Dark venues need a lower L_min because the whole image moves toward black.
RED_LAB_THRESHOLDS = {
    "laser_dark": (8, 255, 18, 127, -45, 95),
    "laser_normal": (15, 255, 18, 127, -40, 90),
    "laser_bright": (30, 255, 25, 127, -30, 80),
    "a4_dark": (4, 255, 10, 127, -55, 105),
}
if WORKFLOW == "a4_tape":
    RED_PROFILE = "a4_dark"
else:
    RED_PROFILE = "laser_dark"
RED_LAB_THRESH = RED_LAB_THRESHOLDS.get(RED_PROFILE, RED_LAB_THRESHOLDS["laser_normal"])
MIN_SPOT_AREA = 3
MAX_SPOT_AREA = 2500
MAX_SPOT_WIDTH = 60
MAX_SPOT_HEIGHT = 60
SHAPE_CHECK_AREA = 150
MIN_FILL_RATIO = 0.55
MIN_ASPECT_RATIO = 0.35
MAX_ASPECT_RATIO = 2.8

# A4 black tape detection.
A4_BLACK_THRESH = (0, 75, -35, 35, -35, 35)
A4_BLACK_MIN_AREA = 80
A4_BLACK_MAX_AREA = 50000
A4_ASPECT_PORTRAIT = 21.0 / 29.7
A4_ASPECT_LANDSCAPE = 29.7 / 21.0
A4_ASPECT_TOL = 0.30
A4_FILL_MIN = 0.02
A4_FILL_MAX = 1.05
A4_TAPE_WIDTH_CM = 1.8
A4_TAPE_TOL_CM = 0.5
A4_FAIL_OFF_CM = 5.0
A4_MIN_RECT_AREA = 1800
A4_MAX_RECT_AREA_RATIO = 0.86
A4_MIN_EDGE_PX = 24
A4_DETECT_EVERY_FRAMES = 3
A4_HOLD_FRAMES = 18
A4_PACKET_EVERY_MS = 300
A4_DIR_MIN_PROGRESS = 0.004
A4_RECT_THRESHOLD = 3500
A4_DEBUG = True

# Calibration workflow.
MODE = "calibrate"  # "calibrate" or "track"
CALIB_FILE = "calibration_points.txt"
CALIB_STEPS = [
    ("TL", -SCREEN_HALF_CM, -SCREEN_HALF_CM),
    ("TR", SCREEN_HALF_CM, -SCREEN_HALF_CM),
    ("BR", SCREEN_HALF_CM, SCREEN_HALF_CM),
    ("BL", -SCREEN_HALF_CM, SCREEN_HALF_CM),
    ("CENTER", 0.0, 0.0),
]
TOUCH_HINT = "Tap screen to confirm current point"

CALIB_EXPECTED_PIXEL = {
    "TL": (IMG_W * 0.25, IMG_H * 0.25),
    "TR": (IMG_W * 0.75, IMG_H * 0.25),
    "BR": (IMG_W * 0.75, IMG_H * 0.75),
    "BL": (IMG_W * 0.25, IMG_H * 0.75),
    "CENTER": (IMG_W * 0.50, IMG_H * 0.50),
}

# Runtime state.
cam = camera.Camera(IMG_W, IMG_H)
profile = CAMERA_PROFILES.get(CAMERA_PROFILE, CAMERA_PROFILES["laser_normal"])
cam.exposure(profile["exposure"])
cam.gain(profile["gain"])
disp = display.Display()
ts = touchscreen.TouchScreen()
try:
    DISP_W = disp.width()
    DISP_H = disp.height()
except:
    DISP_W = IMG_W
    DISP_H = IMG_H

last_tick = time.ticks_ms()
fps = 0
frame_cnt = 0

view_x = None
view_y = None
last_pressed = False
status_msg = ""

H_MATRIX = None
calib_points = {}
calib_index = 0

a4_frame_index = 0
a4_target = None
a4_target_age = A4_HOLD_FRAMES + 1
a4_run_active = False
a4_run_start_ms = 0
a4_lap_progress = 0.0
a4_last_progress = None
a4_last_spot = None
a4_off_move_cm = 0.0
a4_result_msg = "READY"
a4_last_packet_ms = 0
a4_dbg_blobs = 0
a4_dbg_blob_pass = 0
a4_dbg_rects = 0
a4_dbg_reason = "INIT"


def distance_cm(x1, y1, x2, y2):
    dx = x1 - x2
    dy = y1 - y2
    return math.sqrt(dx * dx + dy * dy)


def point_distance_px(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return math.sqrt(dx * dx + dy * dy)


def polygon_area(points):
    if points is None or len(points) < 3:
        return 0.0
    s = 0.0
    last = len(points) - 1
    for i in range(len(points)):
        x1, y1 = points[last]
        x2, y2 = points[i]
        s += x1 * y2 - x2 * y1
        last = i
    return abs(s) * 0.5


def order_rect_corners(points):
    if points is None or len(points) < 4:
        return None

    pts = [(float(p[0]), float(p[1])) for p in points[:4]]
    cx = sum(p[0] for p in pts) / 4.0
    cy = sum(p[1] for p in pts) / 4.0
    pts.sort(key=lambda p: math.atan2(p[1] - cy, p[0] - cx))

    min_idx = 0
    min_sum = pts[0][0] + pts[0][1]
    for i in range(1, 4):
        s = pts[i][0] + pts[i][1]
        if s < min_sum:
            min_sum = s
            min_idx = i
    pts = pts[min_idx:] + pts[:min_idx]

    # Image coordinates grow downward; this order is clockwise on the screen.
    if signed_polygon_area(pts) < 0:
        pts = [pts[0], pts[3], pts[2], pts[1]]
    return pts


def signed_polygon_area(points):
    s = 0.0
    last = len(points) - 1
    for i in range(len(points)):
        x1, y1 = points[last]
        x2, y2 = points[i]
        s += x1 * y2 - x2 * y1
        last = i
    return s * 0.5


def point_in_convex_quad(px, py, quad):
    sign = 0
    for i in range(4):
        x1, y1 = quad[i]
        x2, y2 = quad[(i + 1) % 4]
        cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
        if abs(cross) < 1e-6:
            continue
        now_sign = 1 if cross > 0 else -1
        if sign == 0:
            sign = now_sign
        elif sign != now_sign:
            return False
    return True


def point_segment_distance_px(px, py, ax, ay, bx, by):
    vx = bx - ax
    vy = by - ay
    wx = px - ax
    wy = py - ay
    denom = vx * vx + vy * vy
    if denom <= 1e-6:
        return point_distance_px((px, py), (ax, ay)), 0.0
    t = (wx * vx + wy * vy) / denom
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    qx = ax + t * vx
    qy = ay + t * vy
    return point_distance_px((px, py), (qx, qy)), t


def rect_edge_lengths(rect):
    if rect is None:
        return None
    return [point_distance_px(rect[i], rect[(i + 1) % 4]) for i in range(4)]


def cm_per_px_from_rect(rect):
    lengths = rect_edge_lengths(rect)
    if lengths is None:
        return CM_PER_PIXEL_FALLBACK
    short_px = min(lengths)
    long_px = max(lengths)
    if short_px <= 1.0 or long_px <= 1.0:
        return CM_PER_PIXEL_FALLBACK

    cm_a = 21.0 / short_px
    cm_b = 29.7 / long_px
    return (cm_a + cm_b) * 0.5


def tape_state_for_spot(rect, px, py):
    if rect is None:
        return None

    cm_per_px = cm_per_px_from_rect(rect)
    tape_half_px = (A4_TAPE_WIDTH_CM + A4_TAPE_TOL_CM) / max(cm_per_px, 0.001)
    inside = point_in_convex_quad(px, py, rect)

    best_dist = 99999.0
    best_progress = 0.0
    best_edge = 0
    total_len = 0.0
    edge_lens = rect_edge_lengths(rect)
    if edge_lens is None:
        return None
    perimeter = sum(edge_lens)
    if perimeter <= 1.0:
        return None

    for i in range(4):
        ax, ay = rect[i]
        bx, by = rect[(i + 1) % 4]
        dist, t = point_segment_distance_px(px, py, ax, ay, bx, by)
        if dist < best_dist:
            best_dist = dist
            best_progress = (total_len + edge_lens[i] * t) / perimeter
            best_edge = i
        total_len += edge_lens[i]

    dist_cm = best_dist * cm_per_px
    on_tape = inside and best_dist <= tape_half_px
    return on_tape, dist_cm, best_progress, best_edge, cm_per_px


def solve_linear_system(a, b):
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]

    for col in range(n):
        pivot = col
        pivot_abs = abs(m[col][col])
        for r in range(col + 1, n):
            v = abs(m[r][col])
            if v > pivot_abs:
                pivot = r
                pivot_abs = v
        if pivot_abs < 1e-9:
            return None
        if pivot != col:
            m[col], m[pivot] = m[pivot], m[col]

        div = m[col][col]
        for c in range(col, n + 1):
            m[col][c] /= div

        for r in range(n):
            if r == col:
                continue
            factor = m[r][col]
            if factor == 0:
                continue
            for c in range(col, n + 1):
                m[r][c] -= factor * m[col][c]

    return [m[i][n] for i in range(n)]


def compute_homography(src_pts, dst_pts):
    a = []
    b = []
    for (u, v), (x, y) in zip(src_pts, dst_pts):
        a.append([u, v, 1.0, 0.0, 0.0, 0.0, -u * x, -v * x])
        b.append(x)
        a.append([0.0, 0.0, 0.0, u, v, 1.0, -u * y, -v * y])
        b.append(y)

    h = solve_linear_system(a, b)
    if h is None:
        return None
    return [h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], 1.0]


def apply_homography(h, px, py):
    den = h[6] * px + h[7] * py + h[8]
    if abs(den) < 1e-9:
        return None
    x = (h[0] * px + h[1] * py + h[2]) / den
    y = (h[3] * px + h[4] * py + h[5]) / den
    return x, y


def pixel_to_cm(px, py):
    if H_MATRIX is not None:
        result = apply_homography(H_MATRIX, px, py)
        if result is not None:
            return result
    return (px - ORIGIN_X) * CM_PER_PIXEL_FALLBACK, (py - ORIGIN_Y) * CM_PER_PIXEL_FALLBACK


def is_valid_spot_blob(b):
    area = b.area()
    if area < MIN_SPOT_AREA or area > MAX_SPOT_AREA:
        return False

    w = b.w()
    h = b.h()
    if w == 0 or h == 0:
        return False
    if w > MAX_SPOT_WIDTH or h > MAX_SPOT_HEIGHT:
        return False

    aspect = w / h
    if aspect < MIN_ASPECT_RATIO or aspect > MAX_ASPECT_RATIO:
        return False

    if area > SHAPE_CHECK_AREA and area / (w * h) < MIN_FILL_RATIO:
        return False

    return True


def get_valid_spot_blobs(blobs):
    result = []
    for b in blobs:
        if is_valid_spot_blob(b):
            result.append(b)
    return result


def select_target_blob(candidates, step_label=None):
    best_blob = None
    best_score = -999999.0
    expected = CALIB_EXPECTED_PIXEL.get(step_label)
    for b in candidates:
        area = b.area()
        w = b.w()
        h = b.h()
        aspect = w / h
        aspect_penalty = abs(aspect - 1.0) * 12.0
        size_penalty = max(0, w - 18) + max(0, h - 18)
        region_penalty = 0.0
        if expected is not None:
            dx = b.cx() - expected[0]
            dy = b.cy() - expected[1]
            region_penalty = math.sqrt(dx * dx + dy * dy) * 0.35
        score = area - aspect_penalty - size_penalty - region_penalty
        if score > best_score:
            best_score = score
            best_blob = b
    return best_blob


def detect_red_spot(img, step_label=None):
    blobs = img.find_blobs(
        [RED_LAB_THRESH],
        pixels_threshold=MIN_SPOT_AREA,
        area_threshold=MIN_SPOT_AREA,
        merge=True
    )
    candidates = get_valid_spot_blobs(blobs)
    return select_target_blob(candidates, step_label), candidates


def blob_pixel_count(b):
    try:
        return b.pixels()
    except:
        return b.area()


def normalize_corner_points(raw_points):
    points = []
    for p in raw_points:
        try:
            points.append((float(p[0]), float(p[1])))
            continue
        except:
            pass
        try:
            points.append((float(p.x()), float(p.y())))
        except:
            pass
    if len(points) < 4:
        return None
    return points[:4]


def blob_rect_corners(b):
    for method_name in ("mini_corners", "min_corners", "corners"):
        try:
            method = getattr(b, method_name)
            points = normalize_corner_points(method())
            if points is not None:
                ordered = order_rect_corners(points)
                if ordered is not None:
                    return ordered
        except:
            pass

    return order_rect_corners([
        (b.x(), b.y()),
        (b.x() + b.w(), b.y()),
        (b.x() + b.w(), b.y() + b.h()),
        (b.x(), b.y() + b.h()),
    ])


def score_a4_blob(b):
    global a4_dbg_reason

    pixels = blob_pixel_count(b)
    if pixels < A4_BLACK_MIN_AREA or pixels > A4_BLACK_MAX_AREA:
        a4_dbg_reason = "area {}".format(pixels)
        return None

    rect = blob_rect_corners(b)
    if rect is None:
        a4_dbg_reason = "corner"
        return None

    rect_area = polygon_area(rect)
    if rect_area < A4_MIN_RECT_AREA:
        a4_dbg_reason = "rect {:.0f}".format(rect_area)
        return None
    if rect_area > IMG_W * IMG_H * A4_MAX_RECT_AREA_RATIO:
        a4_dbg_reason = "big {:.0f}".format(rect_area)
        return None

    lengths = rect_edge_lengths(rect)
    if lengths is None:
        a4_dbg_reason = "length"
        return None

    short_px = min(lengths)
    long_px = max(lengths)
    if short_px < A4_MIN_EDGE_PX or long_px <= 1.0:
        a4_dbg_reason = "edge {:.0f}".format(short_px)
        return None

    aspect = short_px / long_px
    if abs(aspect - A4_ASPECT_PORTRAIT) > A4_ASPECT_TOL:
        a4_dbg_reason = "aspect {:.2f}".format(aspect)
        return None

    fill = pixels / rect_area
    if fill < A4_FILL_MIN or fill > A4_FILL_MAX:
        a4_dbg_reason = "fill {:.2f}".format(fill)
        return None

    aspect_score = max(0.0, 1.0 - abs(aspect - A4_ASPECT_PORTRAIT) * 2.5)
    fill_score = max(0.0, 1.0 - abs(fill - 0.27) * 3.0)
    size_score = min(rect_area, 18000.0) / 18000.0
    score = aspect_score * 120.0 + fill_score * 80.0 + size_score * 40.0

    return {
        "rect": rect,
        "score": score,
        "pixels": pixels,
        "area": rect_area,
        "fill": fill,
        "aspect": aspect,
        "cm_per_px": cm_per_px_from_rect(rect),
        "source": "blob",
    }


def detect_a4_target(img):
    global a4_dbg_blobs, a4_dbg_blob_pass, a4_dbg_rects, a4_dbg_reason

    a4_dbg_blobs = 0
    a4_dbg_blob_pass = 0
    a4_dbg_rects = 0
    a4_dbg_reason = "none"

    try:
        blobs = img.find_blobs(
            [A4_BLACK_THRESH],
            pixels_threshold=A4_BLACK_MIN_AREA,
            area_threshold=A4_BLACK_MIN_AREA,
            merge=True
        )
    except:
        return None

    if blobs is None:
        blobs = []

    a4_dbg_blobs = len(blobs)

    best = None
    best_score = -999999.0
    for b in blobs:
        candidate = score_a4_blob(b)
        if candidate is None:
            continue
        a4_dbg_blob_pass += 1
        if candidate["score"] > best_score:
            best_score = candidate["score"]
            best = candidate

    rect_best = detect_a4_rect_target(img)
    if rect_best is not None:
        if best is None or rect_best["score"] > best["score"]:
            best = rect_best

    if best is not None:
        a4_dbg_reason = best["source"]
    return best


def rect_corners_from_obj(r):
    for method_name in ("corners", "rect_corners", "mini_corners"):
        try:
            method = getattr(r, method_name)
            points = normalize_corner_points(method())
            if points is not None:
                return order_rect_corners(points)
        except:
            pass

    try:
        return order_rect_corners([
            (r.x(), r.y()),
            (r.x() + r.w(), r.y()),
            (r.x() + r.w(), r.y() + r.h()),
            (r.x(), r.y() + r.h()),
        ])
    except:
        return None


def score_a4_rect(rect):
    if rect is None:
        return None

    rect_area = polygon_area(rect)
    if rect_area < A4_MIN_RECT_AREA:
        return None
    if rect_area > IMG_W * IMG_H * A4_MAX_RECT_AREA_RATIO:
        return None

    lengths = rect_edge_lengths(rect)
    if lengths is None:
        return None

    short_px = min(lengths)
    long_px = max(lengths)
    if short_px < A4_MIN_EDGE_PX or long_px <= 1.0:
        return None

    aspect = short_px / long_px
    if abs(aspect - A4_ASPECT_PORTRAIT) > A4_ASPECT_TOL:
        return None

    size_score = min(rect_area, 18000.0) / 18000.0
    aspect_score = max(0.0, 1.0 - abs(aspect - A4_ASPECT_PORTRAIT) * 2.5)
    score = 180.0 + aspect_score * 120.0 + size_score * 40.0

    return {
        "rect": rect,
        "score": score,
        "pixels": 0,
        "area": rect_area,
        "fill": 0.0,
        "aspect": aspect,
        "cm_per_px": cm_per_px_from_rect(rect),
        "source": "rect",
    }


def detect_a4_rect_target(img):
    global a4_dbg_rects

    try:
        rects = img.find_rects(threshold=A4_RECT_THRESHOLD)
    except:
        return None

    if rects is None:
        return None

    a4_dbg_rects = len(rects)
    best = None
    best_score = -999999.0
    for r in rects:
        candidate = score_a4_rect(rect_corners_from_obj(r))
        if candidate is None:
            continue
        if candidate["score"] > best_score:
            best_score = candidate["score"]
            best = candidate
    return best


def draw_rect_lines(img, rect, color):
    if rect is None:
        return
    for i in range(4):
        x1, y1 = rect[i]
        x2, y2 = rect[(i + 1) % 4]
        try:
            img.draw_line(int(x1), int(y1), int(x2), int(y2), color)
        except:
            img.draw_rect(int(x1), int(y1), 2, 2, color)


def reset_a4_run(now):
    global a4_run_active, a4_run_start_ms, a4_lap_progress
    global a4_last_progress, a4_last_spot, a4_off_move_cm, a4_result_msg
    global a4_last_packet_ms

    a4_run_active = False
    a4_run_start_ms = now
    a4_lap_progress = 0.0
    a4_last_progress = None
    a4_last_spot = None
    a4_off_move_cm = 0.0
    a4_result_msg = "READY"
    a4_last_packet_ms = now


def load_calibration():
    global H_MATRIX, calib_points

    try:
        with open(CALIB_FILE, "r") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if parts[0] == "H" and len(parts) >= 10:
                    H_MATRIX = [float(x) for x in parts[1:10]]
                elif len(parts) >= 5:
                    label = parts[0]
                    px = float(parts[1])
                    py = float(parts[2])
                    wx = float(parts[3])
                    wy = float(parts[4])
                    calib_points[label] = (px, py, wx, wy)
    except:
        return False

    if H_MATRIX is None:
        corner_labels = ["TL", "TR", "BR", "BL"]
        if all(label in calib_points for label in corner_labels):
            src_pts = [(calib_points[label][0], calib_points[label][1]) for label in corner_labels]
            dst_pts = [(calib_points[label][2], calib_points[label][3]) for label in corner_labels]
            H_MATRIX = compute_homography(src_pts, dst_pts)
    return True


def save_calibration():
    try:
        with open(CALIB_FILE, "w") as f:
            f.write("# label pixel_x pixel_y world_x_cm world_y_cm\n")
            for label, wx, wy in CALIB_STEPS:
                if label in calib_points:
                    px, py, _, _ = calib_points[label]
                    f.write("{} {:.3f} {:.3f} {:.3f} {:.3f}\n".format(label, px, py, wx, wy))
            if H_MATRIX is not None:
                f.write("H {}\n".format(" ".join("{:.8f}".format(v) for v in H_MATRIX)))
        return True
    except:
        return False


def export_calibration_packet():
    parts = []
    for label, wx, wy in CALIB_STEPS:
        if label in calib_points:
            px, py, _, _ = calib_points[label]
            parts.append("{}={:.2f},{:.2f}".format(label, px, py))
    if H_MATRIX is not None:
        parts.append("H=" + ",".join("{:.8f}".format(v) for v in H_MATRIX))
    return "CALIB " + " ".join(parts)


def capture_calibration_point(label, wx, wy, px, py):
    calib_points[label] = (px, py, wx, wy)


def confirm_current_point(raw_x, raw_y):
    global calib_index, H_MATRIX, MODE, status_msg

    if calib_index >= len(CALIB_STEPS):
        return

    label, wx, wy = CALIB_STEPS[calib_index]
    capture_calibration_point(label, wx, wy, raw_x, raw_y)
    save_calibration()
    print("Captured {}: {:.2f}, {:.2f}".format(label, raw_x, raw_y))

    calib_index += 1
    status_msg = "Saved {}".format(label)

    if calib_index >= len(CALIB_STEPS):
        corner_labels = ["TL", "TR", "BR", "BL"]
        src_pts = [(calib_points[label][0], calib_points[label][1]) for label in corner_labels]
        dst_pts = [(calib_points[label][2], calib_points[label][3]) for label in corner_labels]
        H_MATRIX = compute_homography(src_pts, dst_pts)
        save_calibration()
        print(export_calibration_packet())
        MODE = "track"
        status_msg = "Calibration done"


load_calibration()
if WORKFLOW == "screen_reset" and MODE == "calibrate":
    calib_points = {}
    H_MATRIX = None
elif WORKFLOW == "a4_tape":
    MODE = "track"
print("Vision ready, workflow = {}, mode = {}, profile = {}".format(WORKFLOW, MODE, CAMERA_PROFILE))

while True:
    img = cam.read()
    now = time.ticks_ms()
    _touch_x, _touch_y, pressed = ts.read()
    touch_tap = pressed and not last_pressed
    last_pressed = pressed

    if WORKFLOW == "a4_tape":
        a4_frame_index += 1
        if touch_tap:
            reset_a4_run(now)
            a4_run_active = True
            a4_result_msg = "RUN"

        if a4_frame_index % A4_DETECT_EVERY_FRAMES == 0:
            found_a4 = detect_a4_target(img)
            if found_a4 is not None:
                a4_target = found_a4
                a4_target_age = 0
            else:
                a4_target_age += 1
        else:
            a4_target_age += 1

        if a4_target_age > A4_HOLD_FRAMES:
            a4_target = None

        target, candidates = detect_red_spot(img)
        for b in candidates:
            img.draw_rect(b.x(), b.y(), b.w(), b.h(), image.COLOR_YELLOW)

        has_spot = target is not None
        spot_state = None
        raw_x = 0.0
        raw_y = 0.0
        if has_spot:
            raw_x = target.cx()
            raw_y = target.cy()
            img.draw_rect(target.x(), target.y(), target.w(), target.h(), image.COLOR_RED)
            img.draw_cross(int(raw_x), int(raw_y), image.COLOR_RED, 2)
            img.draw_circle(int(raw_x), int(raw_y), 5, image.COLOR_RED)

        if a4_target is not None:
            rect = a4_target["rect"]
            draw_rect_lines(img, rect, image.COLOR_GREEN)
            for p in rect:
                img.draw_cross(int(p[0]), int(p[1]), image.COLOR_GREEN, 1)

            if has_spot:
                spot_state = tape_state_for_spot(rect, raw_x, raw_y)

        if a4_run_active and spot_state is not None and has_spot:
            on_tape, dist_tape_cm, progress, edge_idx, cm_per_px = spot_state

            if a4_last_progress is not None:
                delta = progress - a4_last_progress
                if delta < -0.50:
                    delta += 1.0
                elif delta > 0.50:
                    delta -= 1.0
                if delta > A4_DIR_MIN_PROGRESS:
                    a4_lap_progress += delta
            a4_last_progress = progress

            if on_tape:
                a4_off_move_cm = 0.0
            else:
                if a4_last_spot is not None:
                    a4_off_move_cm += point_distance_px((raw_x, raw_y), a4_last_spot) * cm_per_px
                if a4_off_move_cm >= A4_FAIL_OFF_CM:
                    a4_result_msg = "OFF >5cm"
            a4_last_spot = (raw_x, raw_y)

            elapsed_s = time.ticks_diff(now, a4_run_start_ms) / 1000.0
            if a4_lap_progress >= 0.96 and elapsed_s <= 30.0:
                a4_result_msg = "LAP OK"
            elif elapsed_s > 30.0 and a4_lap_progress < 0.96:
                a4_result_msg = "TIMEOUT"

            if time.ticks_diff(now, a4_last_packet_ms) >= A4_PACKET_EVERY_MS:
                a4_last_packet_ms = now
                print("A4 x={:.1f} y={:.1f} on={} d={:.2f} edge={} prog={:.3f} lap={:.3f}".format(
                    raw_x, raw_y, 1 if on_tape else 0, dist_tape_cm, edge_idx, progress, a4_lap_progress
                ))

        if a4_target is None:
            img.draw_string(4, 4, "A4: NO TARGET", image.COLOR_RED)
            if A4_DEBUG:
                img.draw_string(4, 22, "B:{}/{} R:{} {}".format(
                    a4_dbg_blob_pass, a4_dbg_blobs, a4_dbg_rects, a4_dbg_reason
                ), image.COLOR_WHITE)
            else:
                img.draw_string(4, 22, "Need black tape border", image.COLOR_WHITE)
        else:
            img.draw_string(4, 4, "A4: FOUND {}".format(a4_target["source"]), image.COLOR_GREEN)
            img.draw_string(4, 22, "S:{:.3f} asp:{:.2f}".format(
                a4_target["cm_per_px"], a4_target["aspect"]
            ), image.COLOR_WHITE)

        if not has_spot:
            img.draw_string(4, 40, "Red: NO SPOT", image.COLOR_RED)
        elif spot_state is None:
            img.draw_string(4, 40, "Red: WAIT A4 ({:.0f},{:.0f})".format(raw_x, raw_y), image.COLOR_YELLOW)
        else:
            on_tape, dist_tape_cm, progress, edge_idx, cm_per_px = spot_state
            color = image.COLOR_GREEN if on_tape else image.COLOR_RED
            img.draw_string(4, 40, "Red:{} d:{:.2f}cm".format("ON" if on_tape else "OFF", dist_tape_cm), color)
            img.draw_string(4, 58, "Edge:{} Prog:{:.2f}".format(edge_idx, progress), image.COLOR_WHITE)

        if a4_run_active:
            elapsed_s = time.ticks_diff(now, a4_run_start_ms) / 1000.0
            img.draw_string(4, 76, "T:{:.1f}s Lap:{:.2f}".format(elapsed_s, a4_lap_progress), image.COLOR_WHITE)
            img.draw_string(4, 94, "OffMove:{:.1f}cm {}".format(a4_off_move_cm, a4_result_msg), image.COLOR_YELLOW)
        else:
            img.draw_string(4, 76, "Tap screen to start", image.COLOR_WHITE)

    else:
        # Origin and tolerance box.
        img.draw_cross(ORIGIN_X, ORIGIN_Y, image.COLOR_BLUE, 2)
        img.draw_rect(
            ORIGIN_X - RESET_ERROR_LIMIT_PX,
            ORIGIN_Y - RESET_ERROR_LIMIT_PX,
            RESET_ERROR_LIMIT_PX * 2,
            RESET_ERROR_LIMIT_PX * 2,
            image.COLOR_BLUE
        )

        active_step_label = None
        if MODE == "calibrate" and calib_index < len(CALIB_STEPS):
            active_step_label = CALIB_STEPS[calib_index][0]
        target, candidates = detect_red_spot(img, active_step_label)

        if MODE == "calibrate" and touch_tap and target is None:
            status_msg = "No laser spot"

        has_spot = False
        raw_x = 0.0
        raw_y = 0.0
        show_x = 0.0
        show_y = 0.0
        spot_x_cm = 0.0
        spot_y_cm = 0.0
        dist_origin = 999.0

        for b in candidates:
            img.draw_rect(b.x(), b.y(), b.w(), b.h(), image.COLOR_YELLOW)

        if target is not None:
            raw_x = target.cx()
            raw_y = target.cy()
            has_spot = True

            if view_x is None:
                view_x = raw_x
                view_y = raw_y
            else:
                dx = raw_x - view_x
                dy = raw_y - view_y
                if dx * dx + dy * dy > 14 * 14:
                    view_x = raw_x
                    view_y = raw_y
                else:
                    view_x += (raw_x - view_x) * 0.30
                    view_y += (raw_y - view_y) * 0.30

            show_x = view_x
            show_y = view_y
            spot_x_cm, spot_y_cm = pixel_to_cm(raw_x, raw_y)
            dist_origin = distance_cm(spot_x_cm, spot_y_cm, 0.0, 0.0)

            img.draw_rect(target.x(), target.y(), target.w(), target.h(), image.COLOR_RED)
            img.draw_cross(int(show_x), int(show_y), image.COLOR_RED, 2)

            if MODE == "calibrate" and touch_tap:
                confirm_current_point(raw_x, raw_y)

        else:
            view_x = None
            view_y = None
            if MODE == "calibrate" and touch_tap:
                status_msg = "No laser spot"

        if MODE == "calibrate":
            if calib_index < len(CALIB_STEPS):
                label, wx, wy = CALIB_STEPS[calib_index]
                img.draw_string(4, 4, "CALIB  saved:{}/5".format(calib_index), image.COLOR_YELLOW)
                img.draw_string(4, 22, "Step {}/5: {}".format(calib_index + 1, label), image.COLOR_YELLOW)
                img.draw_string(4, 40, "Target: ({:.1f}, {:.1f})cm".format(wx, wy), image.COLOR_WHITE)
                if has_spot:
                    img.draw_string(4, 58, TOUCH_HINT, image.COLOR_WHITE)
                    img.draw_string(4, 76, "X:{:.1f} Y:{:.1f}cm".format(spot_x_cm, spot_y_cm), image.COLOR_WHITE)
                else:
                    img.draw_string(4, 58, "Find the red spot", image.COLOR_RED)
                if status_msg:
                    img.draw_string(4, 94, status_msg, image.COLOR_GREEN)
            else:
                img.draw_string(4, 4, "Calibration done", image.COLOR_GREEN)
                MODE = "track"

        if MODE == "track":
            if status_msg:
                img.draw_string(4, 58, status_msg, image.COLOR_GREEN)
            if has_spot:
                img.draw_string(4, 4, "X:{:.1f} Y:{:.1f}cm".format(spot_x_cm, spot_y_cm), image.COLOR_WHITE)
                img.draw_string(4, 22, "ERR:{:.2f}cm".format(dist_origin), image.COLOR_WHITE)
                if dist_origin <= RESET_ERROR_LIMIT:
                    img.draw_string(4, 40, "RESET OK", image.COLOR_GREEN)
                else:
                    img.draw_string(4, 40, "RESET FAIL", image.COLOR_RED)
            else:
                img.draw_string(4, 4, "Status: No Spot", image.COLOR_RED)

    frame_cnt += 1
    if time.ticks_diff(now, last_tick) >= 1000:
        fps = frame_cnt
        frame_cnt = 0
        last_tick = now

    img.draw_string(4, 112, "FPS:{} {}".format(fps, CAMERA_PROFILE), image.COLOR_WHITE)
    disp.show(img)
    time.sleep_ms(1)
