from maix import camera, image, display, time, touchscreen
import math

# Camera profiles.
# Switch this line first when the venue light changes.
PROFILE = "laser_dark"  # "laser_dark", "laser_normal", "laser_bright", or "paper"
CAMERA_PROFILES = {
    "laser_dark": {"exposure": 2600, "gain": 65},
    "laser_normal": {"exposure": 1300, "gain": 65},
    "laser_bright": {"exposure": 700, "gain": 50},
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
}
RED_LAB_THRESH = RED_LAB_THRESHOLDS.get(PROFILE, RED_LAB_THRESHOLDS["laser_normal"])
MIN_SPOT_AREA = 5
MAX_SPOT_AREA = 2500
MAX_SPOT_WIDTH = 45
MAX_SPOT_HEIGHT = 45
SHAPE_CHECK_AREA = 150
MIN_FILL_RATIO = 0.55
MIN_ASPECT_RATIO = 0.35
MAX_ASPECT_RATIO = 2.8

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
profile = CAMERA_PROFILES.get(PROFILE, CAMERA_PROFILES["laser_normal"])
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


def distance_cm(x1, y1, x2, y2):
    dx = x1 - x2
    dy = y1 - y2
    return math.sqrt(dx * dx + dy * dy)


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
if MODE == "calibrate":
    calib_points = {}
    H_MATRIX = None
print("Vision ready, mode = {}".format(MODE))

while True:
    img = cam.read()

    # Origin and tolerance box.
    img.draw_cross(ORIGIN_X, ORIGIN_Y, image.COLOR_BLUE, 2)
    img.draw_rect(
        ORIGIN_X - RESET_ERROR_LIMIT_PX,
        ORIGIN_Y - RESET_ERROR_LIMIT_PX,
        RESET_ERROR_LIMIT_PX * 2,
        RESET_ERROR_LIMIT_PX * 2,
        image.COLOR_BLUE
    )

    blobs = img.find_blobs(
        [RED_LAB_THRESH],
        pixels_threshold=MIN_SPOT_AREA,
        area_threshold=MIN_SPOT_AREA,
        merge=True
    )
    candidates = get_valid_spot_blobs(blobs)
    active_step_label = None
    if MODE == "calibrate" and calib_index < len(CALIB_STEPS):
        active_step_label = CALIB_STEPS[calib_index][0]
    target = select_target_blob(candidates, active_step_label)

    _touch_x, _touch_y, pressed = ts.read()
    touch_tap = pressed and not last_pressed
    last_pressed = pressed

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
    now = time.ticks_ms()
    if time.ticks_diff(now, last_tick) >= 1000:
        fps = frame_cnt
        frame_cnt = 0
        last_tick = now

    img.draw_string(4, 112, "FPS:{} {}".format(fps, PROFILE), image.COLOR_WHITE)
    disp.show(img)
    time.sleep_ms(1)
