"""2D 영상 좌표에서만 계산하는 관절 각도. 신뢰도가 부족하면 None을 반환합니다."""

import math

KEYPOINT_NAMES = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)
ANGLE_NAMES = ("left_elbow", "right_elbow", "left_knee", "right_knee",
               "torso_lean", "shoulder_direction", "hip_direction")
SKELETON = ((0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9),
            (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
            (11, 13), (13, 15), (12, 14), (14, 16))


def angle_at(a, b, c):
    """B가 꼭짓점인 0~180도 각도. 겹치는 점/비정상 좌표는 측정 불가."""
    if any(p is None or len(p) != 2 or not all(math.isfinite(v) for v in p)
           for p in (a, b, c)):
        return None
    u = (a[0] - b[0], a[1] - b[1])
    v = (c[0] - b[0], c[1] - b[1])
    denominator = math.hypot(*u) * math.hypot(*v)
    if denominator <= 1e-9:
        return None
    cosine = max(-1.0, min(1.0, (u[0]*v[0] + u[1]*v[1]) / denominator))
    return math.degrees(math.acos(cosine))


def reliable_point(keypoints, index, threshold=0.5):
    """키포인트는 원본 픽셀 x,y,confidence. (0,0)은 미검출로 취급합니다."""
    if keypoints is None or index >= len(keypoints):
        return None
    x, y, confidence = keypoints[index]
    if (not all(math.isfinite(float(v)) for v in (x, y, confidence))
            or confidence < threshold or x < 0 or y < 0 or (x == 0 and y == 0)):
        return None
    return float(x), float(y)


def line_direction(a, b):
    """영상 오른쪽=0도, 아래=+90도인 A→B 방향(-180~180도)."""
    if a is None or b is None or math.dist(a, b) <= 1e-9:
        return None
    return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))


def analyze_pose(keypoints, threshold=0.5):
    points = [reliable_point(keypoints, i, threshold) for i in range(17)]
    angles = {}
    for name, indices in (("left_elbow", (5, 7, 9)), ("right_elbow", (6, 8, 10)),
                          ("left_knee", (11, 13, 15)), ("right_knee", (12, 14, 16))):
        angles[name] = angle_at(*(points[i] for i in indices))
    angles["shoulder_direction"] = line_direction(points[5], points[6])
    angles["hip_direction"] = line_direction(points[11], points[12])
    angles["torso_lean"] = None
    if all(points[i] is not None for i in (5, 6, 11, 12)):
        shoulder = tuple((points[5][j] + points[6][j]) / 2 for j in (0, 1))
        hip = tuple((points[11][j] + points[12][j]) / 2 for j in (0, 1))
        if math.dist(shoulder, hip) > 1e-9:
            # 위쪽 수직=0도, 화면 오른쪽으로 기울면 양수입니다.
            angles["torso_lean"] = math.degrees(math.atan2(shoulder[0] - hip[0],
                                                          hip[1] - shoulder[1]))
    return angles
