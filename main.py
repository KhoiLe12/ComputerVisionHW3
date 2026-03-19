import cv2
import numpy as np
import os

# =============================
# PARAMETERS (TUNE THESE)
# =============================

CANNY_LOW = 50
CANNY_HIGH = 200

HOUGH_RHO = 1
HOUGH_THETA = np.pi / 180
HOUGH_THRESHOLD = 22
HOUGH_MIN_LINE_LENGTH = 21
HOUGH_MAX_LINE_GAP = 30

MERGE_X_TOLERANCE = 28 # Max x-distance between segment midpoints to merge into one line

# =============================
# HELPER FUNCTIONS
# =============================

def region_of_interest(image):
    height, width = image.shape

    mask = np.zeros_like(image)

    # Trapezoid ROI: full width at the bottom, wide opening at the top.
    # This ensures outer lane edges are not clipped by the ROI walls.
    bottom_left  = (0,                  height)
    bottom_right = (width,              height)
    top_right    = (int(width * 0.85),  int(height * 0.55))
    top_left     = (int(width * 0.15),  int(height * 0.55))

    polygon = np.array([[bottom_left, bottom_right, top_right, top_left]], np.int32)

    cv2.fillPoly(mask, polygon, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


def filter_lines(lines):
    """Returns all line segments, filtering out only near-horizontal ones."""
    result = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        if x1 == x2:
            continue
        slope = (y2 - y1) / (x2 - x1)
        if abs(slope) < 0.3:
            continue
        result.append((x1, y1, x2, y2))
    return result



def cluster_and_fit(segments, image_height, x_tolerance):
    """Cluster a list of same-slope segments by x-midpoint and fit one line per cluster."""
    if not segments:
        return []
    segments = sorted(segments, key=lambda s: (s[0] + s[2]) // 2)
    clusters = []
    current = [segments[0]]
    for seg in segments[1:]:
        cx = (seg[0] + seg[2]) // 2
        cluster_cx = sum((s[0] + s[2]) // 2 for s in current) // len(current)
        if abs(cx - cluster_cx) <= x_tolerance:
            current.append(seg)
        else:
            clusters.append(current)
            current = [seg]
    clusters.append(current)

    y_min = int(image_height * 0.55)
    y_max = image_height
    merged = []
    for cluster in clusters:
        pts = [(x, y) for (x1, y1, x2, y2) in cluster for x, y in [(x1, y1), (x2, y2)]]
        ys = [p[1] for p in pts]
        xs = [p[0] for p in pts]
        fit = np.polyfit(ys, xs, 1)
        x_bot = int(np.polyval(fit, y_max))
        x_top = int(np.polyval(fit, y_min))
        merged.append((x_bot, y_max, x_top, y_min))
    return merged


def merge_segments(segments, image_height, x_tolerance=40):
    """Split by slope sign, cluster each group independently, then combine."""
    neg = [(x1,y1,x2,y2) for (x1,y1,x2,y2) in segments if x1 != x2 and (y2-y1)/(x2-x1) < 0]
    pos = [(x1,y1,x2,y2) for (x1,y1,x2,y2) in segments if x1 != x2 and (y2-y1)/(x2-x1) > 0]
    return cluster_and_fit(neg, image_height, x_tolerance) + \
           cluster_and_fit(pos, image_height, x_tolerance)


def draw_lane_lines(image, segments):
    line_image = np.zeros_like(image)
    width = image.shape[1]
    height = image.shape[0]

    merged = merge_segments(segments, height, x_tolerance=MERGE_X_TOLERANCE)

    for (x1, y1, x2, y2) in merged:
        cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Find the merged line whose x-center is closest to the image center
    center_seg = min(merged, key=lambda s: abs((s[0] + s[2]) // 2 - width // 2))
    cx = (center_seg[0] + center_seg[2]) // 2
    cy = (center_seg[1] + center_seg[3]) // 2
    cv2.circle(line_image, (cx, cy), 8, (0, 0, 255), -1)

    combined = cv2.addWeighted(image, 0.8, line_image, 1, 1)
    return combined



# =============================
# MAIN PIPELINE
# =============================

def process_image(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blur, CANNY_LOW, CANNY_HIGH)

    roi = region_of_interest(edges)

    lines = cv2.HoughLinesP(
        roi,
        HOUGH_RHO,
        HOUGH_THETA,
        HOUGH_THRESHOLD,
        minLineLength=HOUGH_MIN_LINE_LENGTH,
        maxLineGap=HOUGH_MAX_LINE_GAP
    )

    if lines is None:
        return image

    segments = filter_lines(lines)

    if not segments:
        return image

    output = draw_lane_lines(image, segments)

    return output


# =============================
# PROCESS ALL IMAGES
# =============================

input_folder = "Images"
output_folder = "output"

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        img_path = os.path.join(input_folder, filename)
        image = cv2.imread(img_path)

        result = process_image(image)

        save_path = os.path.join(output_folder, filename)
        cv2.imwrite(save_path, result)

print("Processing complete!")