"""야구 영상에서 사람과 2D 관절 자세를 표시합니다. 실행: python main.py"""

import math
import csv
from datetime import datetime
import os
from pathlib import Path
import sys
import tempfile
import time
from functools import lru_cache

from pose_utils import (KEYPOINT_NAMES, ANGLE_NAMES, SKELETON,
                        reliable_point, analyze_pose)


WINDOW_NAME = "Baseball AI Detector"
PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "yolo11n-pose.pt"
CONFIDENCE = 0.25
KEYPOINT_CONFIDENCE = 0.5


def select_video():
    # Tkinter는 파일 선택과 화면 크기 확인에만 사용합니다.
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError("파일 선택창을 열 수 없습니다. Python 설치에서 Tcl/Tk 및 "
                           f"Tkinter 구성 요소를 확인해 주세요. 원인: {exc}") from exc
    except Exception as exc:
        raise RuntimeError("파일 선택창 초기화에 실패했습니다. Python의 Tcl/Tk 설치와 "
                           f"화면 접근 권한을 확인해 주세요. 원인: {exc}") from exc
    root.withdraw()
    try:
        size = (max(320, root.winfo_screenwidth() - 100),
                max(240, root.winfo_screenheight() - 160))
        filename = filedialog.askopenfilename(
            parent=root,
            title="분석할 야구 영상을 선택하세요",
            filetypes=[("영상 파일", "*.mp4 *.avi *.mov *.mkv *.wmv *.m4v"),
                       ("모든 파일", "*.*")],
        )
        return filename, size
    finally:
        root.destroy()


def open_video(filename):
    if not Path(filename).is_file():
        raise RuntimeError("영상 파일이 존재하지 않습니다. 파일 위치를 확인해 주세요.")
    capture = cv2.VideoCapture(str(filename))
    try:
        if not capture.isOpened():
            raise RuntimeError("영상을 열 수 없습니다. 파일 손상 또는 지원하지 않는 코덱일 수 있습니다.")
        ok, frame = capture.read()
        if not ok or frame is None or frame.size == 0:
            raise RuntimeError("첫 프레임을 읽지 못했습니다. 영상 손상이나 코덱 문제를 확인해 주세요.")
        return capture, frame
    except Exception:
        capture.release()
        raise


def load_model():
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not MODEL_PATH.is_file():
        print("YOLO11n Pose 모델이 없습니다. Ultralytics 공식 다운로드를 시작합니다.", flush=True)
    try:
        # 절대 경로를 전달하면 공식 자동 다운로드도 models 폴더에 저장됩니다.
        return YOLO(str(MODEL_PATH), task="pose")
    except Exception as exc:
        raise RuntimeError(
            "YOLO 모델 로딩/다운로드에 실패했습니다. 인터넷 연결, 방화벽, "
            "models 폴더 쓰기 권한을 확인해 주세요. 기존 모델이 있다면 손상 여부도 확인해 주세요. "
            f"원인: {exc}"
        ) from exc


def detect_people(model, frame):
    # COCO의 0번 클래스(person)만 추론하며 CPU에서 실행합니다.
    result = model.predict(source=frame, classes=[0], conf=CONFIDENCE,
                           device="cpu", imgsz=640, verbose=False,
                           save=False, show=False)[0]
    if result.boxes is None or len(result.boxes) == 0:
        return []
    coordinates = result.boxes.xyxy.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()
    keypoints = (result.keypoints.data.cpu().numpy()
                 if result.keypoints is not None else None)
    height, width = frame.shape[:2]
    people = []
    for index, (coords, confidence) in enumerate(zip(coordinates, confidences)):
        # 화면 바깥 좌표나 비정상 박스를 걸러 OpenCV에 안전하게 전달합니다.
        if not np.isfinite(coords).all() or not math.isfinite(float(confidence)):
            continue
        x1, y1, x2, y2 = coords
        x1 = int(np.clip(x1, 0, width - 1))
        y1 = int(np.clip(y1, 0, height - 1))
        x2 = int(np.clip(x2, 0, width - 1))
        y2 = int(np.clip(y2, 0, height - 1))
        if x2 > x1 and y2 > y1:
            points = np.zeros((17, 3), dtype=float)
            if keypoints is not None and index < len(keypoints) and keypoints[index].shape == (17, 3):
                points = keypoints[index].copy()
                # 화면 밖 좌표는 경계로 끌어오지 않고 신뢰도를 0으로 만듭니다.
                invalid = (~np.isfinite(points).all(axis=1) | (points[:, 0] < 0)
                           | (points[:, 0] >= width) | (points[:, 1] < 0)
                           | (points[:, 1] >= height))
                points[invalid] = 0
            people.append({"box": (x1, y1, x2, y2), "confidence": float(confidence),
                           "keypoints": points})
    return people


def box_iou(a, b):
    intersection = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    area = lambda box: (box[2]-box[0]) * (box[3]-box[1])
    return intersection / max(1, area(a) + area(b) - intersection)


def choose_target(people, previous_box, frame_shape):
    if not people:
        return None
    if previous_box is None:
        height, width = frame_shape[:2]
        def score(person):
            x1, y1, x2, y2 = person["box"]
            centrality = 1 - abs((x1+x2)/2 - width/2) / width
            visibility = sum(reliable_point(person["keypoints"], i, KEYPOINT_CONFIDENCE)
                             is not None for i in range(5, 17)) / 12
            return (x2-x1)*(y2-y1) * centrality * (0.5 + visibility)
        return max(range(len(people)), key=lambda i: score(people[i]))
    # 위치 연속성으로 연결하는 간단한 추적이며 신원 인식을 보장하지 않습니다.
    px1, py1, px2, py2 = previous_box
    diagonal = math.hypot(px2-px1, py2-py1)
    matches = []
    for index, person in enumerate(people):
        box = person["box"]
        distance = math.hypot((box[0]+box[2]-px1-px2)/2,
                              (box[1]+box[3]-py1-py2)/2) / max(1, diagonal)
        ratio = ((box[2]-box[0])*(box[3]-box[1])) / max(1, (px2-px1)*(py2-py1))
        overlap = box_iou(previous_box, box)
        if 0.4 <= ratio <= 2.5 and (overlap >= 0.1 or distance <= 0.45):
            matches.append((overlap - distance, index))
    return max(matches)[1] if matches else None


@lru_cache(maxsize=1)
def display_font():
    from PIL import ImageFont
    font = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "malgun.ttf"
    return ImageFont.truetype(str(font), 17) if font.is_file() else ImageFont.load_default()


def view_size(frame, screen_size):
    height, width = frame.shape[:2]
    panel_width = 330
    scale = min(1.0, max(160, screen_size[0]-panel_width)/width,
                min(720, screen_size[1])/height, 1100/width)
    return max(1, round(width*scale)), max(1, round(height*scale)), panel_width


def draw_frame(frame, people, fps, paused, screen_size, selected=None,
               recording=False, ended=False, frame_number=0):
    from PIL import Image, ImageDraw
    height, width = frame.shape[:2]
    vw, vh, panel_width = view_size(frame, screen_size)
    canvas_height, canvas_width = max(vh, 460), vw+panel_width
    # MP4 인코더가 홀수 크기를 잘라내지 않도록 가장자리만 1픽셀 보완합니다.
    image = np.zeros((canvas_height + canvas_height % 2,
                      canvas_width + canvas_width % 2, 3), dtype=np.uint8)
    image[:vh, :vw] = cv2.resize(frame, (vw, vh))
    sx, sy = vw/width, vh/height
    for index, person in enumerate(people):
        x1, y1, x2, y2 = person["box"]
        cv2.rectangle(image, (round(x1*sx), round(y1*sy)),
                      (round(x2*sx), round(y2*sy)), (0, 255, 0), 3 if index == selected else 1)
        cv2.putText(image, f"{'BATTER' if index == selected else 'person'} {person['confidence']:.2f}",
                    (round(x1*sx), max(16, round(y1*sy)-7)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        points = [reliable_point(person["keypoints"], i, KEYPOINT_CONFIDENCE) for i in range(17)]
        for a, b in SKELETON:
            if points[a] is not None and points[b] is not None:
                cv2.line(image, (round(points[a][0]*sx), round(points[a][1]*sy)),
                         (round(points[b][0]*sx), round(points[b][1]*sy)), (255, 200, 0), 2, cv2.LINE_AA)
        for point in points:
            if point is not None:
                cv2.circle(image, (round(point[0]*sx), round(point[1]*sy)),
                           4, (0, 180, 255), -1, cv2.LINE_AA)
    target = people[selected] if selected is not None else None
    angles = analyze_pose(target["keypoints"] if target else None, KEYPOINT_CONFIDENCE)
    labels = ("왼쪽 팔꿈치", "오른쪽 팔꿈치", "왼쪽 무릎", "오른쪽 무릎",
              "상체 기울기", "어깨선 방향", "골반선 방향")
    lines = [f"FPS: {fps:.1f} | 사람: {len(people)}", f"프레임: {frame_number}",
             "타자 선택됨" if target else "타자 미검출 / 클릭해서 지정"]
    for key, label in zip(ANGLE_NAMES, labels):
        value = angles[key]
        lines.append(f"{label}: " + (f"{value:.1f}°" if value is not None else "측정 불가"))
    lines += ["2D 영상상 각도 (3D 회전 아님)", "클릭: 타자 선택", "Q: 종료 / Space: 일시정지",
              "R: 처음부터 / S: 영상 저장", "C: 현재까지 CSV 저장",
              "REC: 저장 중" if recording else "영상 저장 꺼짐"]
    if ended:
        lines.append("영상 끝: C 저장 / R 재생 / Q 종료")
    elif paused:
        lines.append("일시정지")
    # Windows 맑은 고딕으로 '측정 불가'를 포함한 한글을 표시합니다.
    pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    drawing = ImageDraw.Draw(pil)
    for index, text in enumerate(lines):
        drawing.text((vw+10, 10+index*25), text, font=display_font(), fill=(240, 240, 240))
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def analysis_row(number, seconds, person, target_id):
    row = {"frame": number, "video_time_s": seconds, "target_id": target_id,
           "person_confidence": person["confidence"] if person else None}
    points = person["keypoints"] if person else None
    for index, name in enumerate(KEYPOINT_NAMES):
        point = reliable_point(points, index, KEYPOINT_CONFIDENCE)
        row[f"{name}_x"] = point[0] if point else None
        row[f"{name}_y"] = point[1] if point else None
        confidence = float(points[index][2]) if points is not None else None
        row[f"{name}_confidence"] = confidence if confidence is not None and math.isfinite(confidence) else None
    row.update(analyze_pose(points, KEYPOINT_CONFIDENCE))
    return row


def output_path(suffix):
    directory = PROJECT_DIR / "outputs"
    directory.mkdir(exist_ok=True)
    return directory / (datetime.now().strftime("analysis_%Y%m%d_%H%M%S_%f") + suffix)


def save_csv(rows):
    if not rows:
        print("저장할 분석 데이터가 없습니다.")
        return None
    path = output_path(".csv")
    try:
        with path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(next(iter(rows.values()))))
            writer.writeheader()
            writer.writerows(rows.values())
    except Exception:
        path.unlink(missing_ok=True)
        raise
    print(f"CSV 저장: {path}")
    return path


class VideoRecorder:
    def __init__(self):
        self.writer = None
        self.path = None

    def start(self, image, fps):
        self.path = output_path(".mp4")
        try:
            self.writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"),
                                          fps, (image.shape[1], image.shape[0]))
            if not self.writer.isOpened():
                raise RuntimeError("MP4 인코더를 열 수 없습니다. 코덱/저장 권한을 확인해 주세요.")
        except Exception:
            self.stop()
            self.path.unlink(missing_ok=True)
            raise
        print(f"분석 영상 저장 시작: {self.path}")

    def write(self, image):
        if self.writer is not None:
            self.writer.write(image)

    def stop(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None
            print(f"분석 영상 저장 종료: {self.path}")


def play_video(filename, screen_size):
    capture = None
    recorder = VideoRecorder()
    try:
        capture, frame = open_video(filename)
        print("YOLO11n Pose 모델을 불러오는 중입니다. CPU를 사용합니다.", flush=True)
        model = load_model()
        source_fps = capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(source_fps) or source_fps <= 0:
            source_fps = 30.0
        frame_interval = 1.0 / source_fps
        total_frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        processed = 0
        paused = False
        people = []
        fps = 0.0
        selected = None
        previous_box = None
        target_id = 1
        rows = {}
        ended = False
        needs_analysis = True
        video_time = 0.0
        pending_click = []
        def on_mouse(event, x, y, flags, userdata):
            if event == cv2.EVENT_LBUTTONDOWN:
                pending_click[:] = [(x, y)]
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WINDOW_NAME, on_mouse)
        print("Q: 종료 | Space: 일시정지/재생 | R: 처음부터 | S: 영상 저장 시작/중지 | C: CSV 저장")
        print("사람 박스를 클릭하면 타자를 선택합니다. 영상 끝에서는 C/R/Q 입력을 기다립니다.")
        while True:
            started = time.perf_counter()
            new_frame = False
            if needs_analysis:
                if frame is None:
                    ok, frame = capture.read()
                    if not ok or frame is None or frame.size == 0:
                        # 컨테이너의 프레임 수는 코덱에 따라 부정확할 수 있습니다.
                        if math.isfinite(total_frames) and total_frames > processed + 1:
                            print(f"영상 프레임 읽기 실패: {processed}프레임에서 중단되었습니다. "
                                  "파일 손상/코덱 문제 또는 부정확한 영상 길이 정보일 수 있습니다.")
                            return 1
                        recorder.stop()
                        print("영상 끝입니다. C: CSV 저장 | R: 다시 재생 | Q: 종료")
                        frame = last_frame
                        ended, paused, needs_analysis = True, True, False
                if needs_analysis:
                    people = detect_people(model, frame)
                    selected = choose_target(people, previous_box, frame.shape)
                    if selected is not None:
                        previous_box = people[selected]["box"]
                    processed += 1
                    timestamp = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                    if math.isfinite(timestamp) and timestamp >= 0 and (processed == 1 or timestamp > video_time):
                        video_time = timestamp
                    else:
                        video_time = max(video_time, (processed-1)/source_fps)
                    last_frame = frame
                    new_frame = True
                    needs_analysis = False
            if pending_click:
                x, y = pending_click.pop()
                vw, vh, _ = view_size(frame, screen_size)
                if 0 <= x < vw and 0 <= y < vh:
                    x, y = x*frame.shape[1]/vw, y*frame.shape[0]/vh
                    hits = [i for i, p in enumerate(people) if
                            p["box"][0] <= x <= p["box"][2] and p["box"][1] <= y <= p["box"][3]]
                    if hits:
                        selected = min(hits, key=lambda i: (people[i]["box"][2]-people[i]["box"][0]) *
                                       (people[i]["box"][3]-people[i]["box"][1]))
                        previous_box = people[selected]["box"]
                        target_id += 1
            # 일시정지/클릭 시 같은 프레임 행을 갱신하며 중복 행을 만들지 않습니다.
            rows[processed] = analysis_row(processed, video_time,
                                           people[selected] if selected is not None else None, target_id)
            displayed = draw_frame(frame, people, fps, paused, screen_size, selected,
                                   recorder.writer is not None, ended, processed)
            if new_frame:
                try:
                    recorder.write(displayed)
                except cv2.error as exc:
                    recorder.stop()
                    print(f"영상 기록 중 오류가 발생해 저장을 중지했습니다. 원인: {exc}")
            cv2.imshow(WINDOW_NAME, displayed)
            remaining = frame_interval - (time.perf_counter() - started)
            delay = 30 if paused else max(1, min(1000, round(remaining * 1000)))
            key = cv2.waitKey(delay) & 0xFF
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                return 0
            if key in (ord("q"), ord("Q")):
                return 0
            if key in (ord("r"), ord("R")):
                # seek를 지원하지 않는 코덱에서도 확실히 처음으로 돌아갑니다.
                capture.release()
                recorder.stop()
                capture, frame = open_video(filename)
                processed, fps, paused = 0, 0.0, False
                people, selected, previous_box = [], None, None
                rows.clear()
                pending_click.clear()
                target_id = 1
                video_time = 0.0
                ended, needs_analysis = False, True
                continue
            if new_frame:
                fps = 1.0 / max(time.perf_counter() - started, 1e-9)
            if key in (ord("c"), ord("C")):
                try:
                    save_csv(rows)
                except (OSError, csv.Error) as exc:
                    print(f"CSV 저장 실패: 저장 경로와 권한을 확인해 주세요. 원인: {exc}")
            if key in (ord("s"), ord("S")):
                if recorder.writer is not None:
                    recorder.stop()
                elif not ended:
                    try:
                        recorder.start(displayed, source_fps)
                        recorder.write(draw_frame(frame, people, fps, paused, screen_size,
                                                   selected, True, ended, processed))
                    except (OSError, RuntimeError, cv2.error) as exc:
                        print(f"분석 영상 저장 실패: {exc}")
                else:
                    print("영상이 끝났습니다. R로 재생한 뒤 S를 눌러 주세요.")
            if key == ord(" ") and not ended:
                paused = not paused
            if not paused and not pending_click:
                frame = None
                needs_analysis = True
    finally:
        recorder.stop()
        if capture is not None:
            capture.release()
        cv2.destroyAllWindows()


def main():
    try:
        filename, screen_size = select_video()
        if not filename:
            print("영상을 선택하지 않아 종료합니다.")
            return 0
        # 취소 시에는 무거운 라이브러리나 모델을 불러오지 않습니다.
        global cv2, np, YOLO
        try:
            import cv2
            import numpy as np
            import PIL  # 한글 각도 패널 표시용 Pillow
            # Ultralytics 설정 파일은 임시 경로에 두며 결과 폴더는 생성하지 않습니다.
            os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(tempfile.gettempdir()) / "baseball-ai-yolo"))
            from ultralytics import YOLO
        except (ImportError, OSError) as exc:
            raise RuntimeError("필수 라이브러리를 불러오지 못했습니다. "
                               "python -m pip install -r requirements.txt 를 실행해 주세요. "
                               f"원인: {exc}") from exc
        return play_video(filename, screen_size)
    except KeyboardInterrupt:
        print("사용자가 실행을 중단했습니다.")
        return 0
    except Exception as exc:
        print(f"오류: {exc}", file=sys.stderr)
        print("영상은 MP4(H.264)로 변환해 다시 시도할 수 있습니다. "
              "OpenCV 창 오류라면 opencv-python-headless 대신 opencv-python 설치를 확인해 주세요.",
              file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
