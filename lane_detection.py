import cv2
import numpy as np
from collections import deque

DEVIATION_FRAMES   = 6     
SMOOTHING_FRAMES   = 7     
SLOPE_MIN_ABS      = 0.35  
DEVIATION_RATIO    = 0.18  
ROI_TOP_FRACTION   = 0.60  

def canny_edge(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    return edges

def region_of_interest(image):
    h, w = image.shape[:2]
    poly = np.array([[
        (int(0.10*w), h),
        (int(0.90*w), h),
        (int(0.58*w), int(ROI_TOP_FRACTION*h)),
        (int(0.42*w), int(ROI_TOP_FRACTION*h))
    ]], dtype=np.int32)
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, poly, 255)
    return cv2.bitwise_and(image, mask)

def line_params(x1, y1, x2, y2):
    if x2 == x1:
        return None
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return m, b

def x_at_y(m, b, y):
    if m == 0:
        return None
    return (y - b) / m

def average_lane_lines(h, lines):
    if lines is None or len(lines) == 0:
        return None, None

    y1_draw = h
    y2_draw = int(h * ROI_TOP_FRACTION)

    left_params, left_w = [], []
    right_params, right_w = [], []

    for x1, y1, x2, y2 in lines:
        lp = line_params(x1, y1, x2, y2)
        if lp is None:
            continue
        m, b = lp
        if abs(m) < SLOPE_MIN_ABS:
            continue
        w = float(np.hypot(x2 - x1, y2 - y1))
        if m < 0:
            left_params.append((m, b))
            left_w.append(w)
        else:
            right_params.append((m, b))
            right_w.append(w)

    def wavg(params, weights):
        if len(params) == 0:
            return None
        params = np.array(params, dtype=np.float64)
        weights = np.array(weights, dtype=np.float64)
        m = np.average(params[:, 0], weights=weights)
        b = np.average(params[:, 1], weights=weights)
        return m, b

    left_mb  = wavg(left_params, left_w)
    right_mb = wavg(right_params, right_w)

    def build_points(mb):
        if mb is None:
            return None
        m, b = mb
        x1d = int(x_at_y(m, b, y1_draw))
        x2d = int(x_at_y(m, b, y2_draw))
        return [x1d, y1_draw, x2d, y2_draw]

    return build_points(left_mb), build_points(right_mb)

def draw_lines(img, lines, color=(0, 255, 0), thickness=6):
    overlay = np.zeros_like(img)
    for line in lines:
        if line is None or len(line) != 4:
            continue
        x1, y1, x2, y2 = map(int, line)
        cv2.line(overlay, (x1, y1), (x2, y2), color, thickness)
    return overlay

# ---------------- MAIN ----------------
input_file = "WhatsApp Video 2025-09-03 at 10.53.53_c1bdb7cf.mp4"
cap = cv2.VideoCapture(input_file)

fps = cap.get(cv2.CAP_PROP_FPS)
if not fps or fps <= 0:
    fps = 25.0  # fallback

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_color = cv2.VideoWriter("lane_detected_output.mp4", fourcc, fps, (width, height))
out_bw    = cv2.VideoWriter("lane_edges_output.mp4", fourcc, fps, (width, height), isColor=False)

lane_centers = deque(maxlen=SMOOTHING_FRAMES)
warning_count = 0
last_left, last_right = None, None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    edges = canny_edge(frame)
    roi = region_of_interest(edges)

    lines = cv2.HoughLinesP(
        roi, 1, np.pi/180, 60,
        minLineLength=int(width * 0.05),
        maxLineGap=int(width * 0.02)
    )
    lines = lines.reshape(-1, 4) if lines is not None else []

    left_line, right_line = average_lane_lines(height, lines)

    if left_line is None and last_left is not None:
        left_line = last_left
    if right_line is None and last_right is not None:
        right_line = last_right

    lane_overlay = draw_lines(frame, [left_line, right_line])
    combo = cv2.addWeighted(frame, 0.85, lane_overlay, 1.0, 0)

    # ---------------- Lane Departure ----------------
    direction = None
    if left_line is not None and right_line is not None:
        y_eval = int(height * 0.90)

        def x_from_line(line):
            x1, y1, x2, y2 = line
            mb = line_params(x1, y1, x2, y2)
            if mb is None:
                return None
            m, b = mb
            return x_at_y(m, b, y_eval)

        x_left  = x_from_line(left_line)
        x_right = x_from_line(right_line)

        if x_left is not None and x_right is not None and x_right > x_left:
            lane_center = 0.5 * (x_left + x_right)
            lane_width  = (x_right - x_left)

            lane_centers.append(lane_center)
            smoothed_center = np.mean(lane_centers)

            vehicle_center = width / 2.0
            offset = vehicle_center - smoothed_center
            threshold_px = DEVIATION_RATIO * lane_width

            if abs(offset) > threshold_px:
                warning_count += 1
                direction = "Right" if offset > 0 else "Left"
            else:
                warning_count = 0

            if warning_count >= DEVIATION_FRAMES:
                cv2.putText(combo, f"⚠ Lane Departure {direction}!",
                            (40, 90), cv2.FONT_HERSHEY_SIMPLEX,
                            1.1, (0, 0, 255), 3)
                print(f"! Lane Departure Detected {direction}!")
        else:
            warning_count = 0
            lane_centers.clear()
    else:
        warning_count = 0
        lane_centers.clear()

    if left_line is not None:
        last_left = left_line
    if right_line is not None:
        last_right = right_line

    # --- Save outputs ---
    out_color.write(combo)
    out_bw.write(roi)

    # --- Show windows ---
    cv2.imshow("Lane Detection", combo)
    cv2.imshow("Edges + ROI (B/W)", roi)

    if cv2.waitKey(int(1000/fps)) & 0xFF == ord('q'):
        break

cap.release()
out_color.release()
out_bw.release()
cv2.destroyAllWindows()
print("✅ Finished lane detection video with both color + black & white outputs")