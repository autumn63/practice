<<<<<<< HEAD
"""
videos.py

필요 패키지:pip install opencv-python
"""

import cv2
from pathlib import Path


# -----------------------------
# 1. 공용 유틸 함수들
# -----------------------------

def open_video_source(video_source):
    """
    영상 입력 소스를 열어서 VideoCapture 객체를 반환하는 함수.

    Parameters
    ----------
    video_source : int or str
        - 0, 1, 2 ... : 웹캠 번호 (보통 0이 기본 웹캠)
        - "cctv.mp4" 같은 문자열 : 영상 파일 경로
        - "rtsp://..." : CCTV RTSP 주소

    Returns
    -------
    video_capture : cv2.VideoCapture
        영상 프레임을 순서대로 읽을 수 있는 OpenCV 객체.
    """
    video_capture = cv2.VideoCapture(video_source)

    if not video_capture.isOpened():
        raise RuntimeError(f"영상 소스를 열 수 없습니다: {video_source}")

    return video_capture


def create_video_writer(output_video_path, frames_per_second, frame_width, frame_height):
    """
    새로운 영상 파일을 저장하기 위한 VideoWriter 객체를 만들어주는 함수.

    Parameters
    ----------
    output_video_path : str or Path
        저장할 출력 영상 파일 경로. (예: 'motion_only.mp4')
    frames_per_second : float
        출력 영상의 FPS 값 (보통 입력 영상의 FPS를 그대로 사용)
    frame_width : int
        영상 가로 길이 (픽셀 수)
    frame_height : int
        영상 세로 길이 (픽셀 수)

    Returns
    -------
    video_writer : cv2.VideoWriter
        frame 단위로 write()를 호출해서 영상으로 저장할 수 있는 객체.
    """
    output_video_path = Path(output_video_path)
    output_video_path.parent.mkdir(parents=True, exist_ok=True)

    # fourcc 는 "어떤 코덱으로 인코딩할지"를 뜻함.
    # "mp4v" 는 mp4 파일을 만들 때 많이 사용하는 기본 코덱 중 하나.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    video_writer = cv2.VideoWriter(
        str(output_video_path),
        fourcc,
        frames_per_second,
        (frame_width, frame_height)
    )

    if not video_writer.isOpened():
        raise RuntimeError(f"영상 파일을 만들 수 없습니다: {output_video_path}")

    return video_writer


# -----------------------------
# 2. 사람(보행자) 검출 설정
# -----------------------------

# HOG + SVM 기반 사람 검출기 (OpenCV에서 기본 제공하는 보행자 detector)
people_detector = cv2.HOGDescriptor()
people_detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


def detect_people_in_frame(color_frame):
    """
    한 프레임(컬러 영상)에서 사람을 찾아내는 함수.

    Parameters
    ----------
    color_frame : numpy.ndarray
        BGR 색상 채널 순서의 영상 프레임 (cv2.VideoCapture로 읽어온 그대로)

    Returns
    -------
    person_boxes : list of tuple
        (x, y, w, h) 형태의 bounding box 리스트.
        x, y 는 좌상단 좌표, w, h 는 너비와 높이.
    person_count : int
        검출된 사람 수.
    """
    # HOG detector는 컬러 그대로 넣어도 동작하지만
    # 보통 크기를 조금 줄여서 속도를 올리기도 한다.
    # 여기서는 이해하기 쉽게 그대로 사용.
    found_rectangles, found_weights = people_detector.detectMultiScale(
        color_frame,
        winStride=(8, 8),   # 슬라이딩 윈도우 이동 간격
        padding=(8, 8),     # 주변 패딩
        scale=1.05          # 이미지 피라미드 스케일
    )

    person_boxes = []
    for (x, y, w, h) in found_rectangles:
        person_boxes.append((x, y, w, h))

    person_count = len(person_boxes)
    return person_boxes, person_count


# -----------------------------
# 3. CCTV 영상에서 사람 유무 + 움직임 구간 추출
# -----------------------------

def extract_motion_video_from_cctv(
    input_video_source,
    output_motion_video_path,
    motion_pixel_ratio_threshold=0.02,
    frame_difference_threshold=25,
    tail_frames_after_motion=15
):
    """
    CCTV 영상에서 "사람이 있는" 프레임만 골라서
    하나의 새로운 영상으로 저장하는 함수.

    기본 아이디어:
    - 연속된 두 프레임의 차이를 보고 "얼마나 많이 변했는지" 계산
    - 프레임의 일정 비율 이상 픽셀이 변하면 → 움직임이 있다고 판단
    - 움직임이 감지된 프레임만 새 VideoWriter로 write()

    Parameters
    ----------
    input_video_source : int or str
        영상 입력 소스 (웹캠 번호, 파일 경로, RTSP 주소 등)
    output_motion_video_path : str or Path
        움직임이 있는 부분만 모아서 저장할 출력 mp4 경로
    motion_pixel_ratio_threshold : float
        0 ~ 1 사이 값. 전체 픽셀 중 이 비율 이상이 변하면
        "움직임 있음"이라고 판단.
        예: 0.02 = 전체 픽셀의 2% 이상이 변한 경우.
    frame_difference_threshold : int
        0 ~ 255 사이 값. 프레임 간 차이에서 이 값 이상일 때
        "변한 픽셀"로 간주.
    tail_frames_after_motion : int
        움직임이 멈춘 뒤에도 몇 프레임 더 이어서 저장할지.
        너무 딱 끊어지지 않게 약간의 여유를 줌.

    Returns
    -------
    motion_segments : list of tuple
        (start_frame_index, end_frame_index) 형태의 리스트.
        움직임이 있었던 구간들의 프레임 인덱스 범위.
    """
    # 1) 입력 영상 열기
    video_capture = open_video_source(input_video_source)

    frames_per_second = video_capture.get(cv2.CAP_PROP_FPS)
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if frames_per_second == 0:
        video_capture.release()
        raise ValueError("입력 영상의 FPS 정보를 읽을 수 없습니다.")

    # 2) 출력용 VideoWriter 생성
    motion_video_writer = create_video_writer(
        output_motion_video_path,
        frames_per_second,
        frame_width,
        frame_height
    )

    # 3) 첫 프레임을 읽어서 "이전 프레임" 기준으로 사용
    has_frame, previous_color_frame = video_capture.read()
    if not has_frame:
        video_capture.release()
        motion_video_writer.release()
        raise ValueError("입력 영상에서 첫 프레임을 읽을 수 없습니다.")

    # 프레임을 흑백으로 바꾸고, 가우시안 블러를 걸어서 노이즈 감소
    previous_gray_frame = cv2.cvtColor(previous_color_frame, cv2.COLOR_BGR2GRAY)
    previous_gray_frame = cv2.GaussianBlur(previous_gray_frame, (5, 5), 0)

    total_pixels = previous_gray_frame.shape[0] * previous_gray_frame.shape[1]

    # 움직임 구간 정보를 저장하기 위한 변수들
    is_currently_in_motion = False          # 지금이 "움직임 구간" 안인지 여부
    remaining_tail_frames = 0               # 움직임 종료 후 tail로 남은 프레임 수
    motion_segments = []                    # (start_frame, end_frame) 리스트
    current_segment_start_index = None      # 현재 구간의 시작 프레임 인덱스

    current_frame_index = 1  # 위에서 이미 첫 프레임을 읽었으므로 1부터 시작

    # 4) 나머지 프레임들 반복 처리
    while True:
        has_frame, current_color_frame = video_capture.read()
        if not has_frame:
            break  # 더 이상 읽을 프레임이 없으면 반복 종료

        # 현재 프레임도 흑백 + 블러 처리
        current_gray_frame = cv2.cvtColor(current_color_frame, cv2.COLOR_BGR2GRAY)
        current_gray_frame = cv2.GaussianBlur(current_gray_frame, (5, 5), 0)

        # 이전 프레임과 현재 프레임의 차이 계산
        frame_difference = cv2.absdiff(previous_gray_frame, current_gray_frame)

        # frame_difference_threshold 보다 큰 픽셀만 남기기 (이진 이미지)
        # → 차이가 작으면 0, 크면 255 로 처리됨
        _, difference_binary = cv2.threshold(
            frame_difference,
            frame_difference_threshold,
            255,
            cv2.THRESH_BINARY
        )

        # 변한 픽셀 개수 세기
        changed_pixel_count = cv2.countNonZero(difference_binary)

        # 전체 픽셀 대비 변한 픽셀 비율 (0 ~ 1)
        changed_pixel_ratio = changed_pixel_count / float(total_pixels)

        # 움직임 여부 판단
        if changed_pixel_ratio > motion_pixel_ratio_threshold:
            # 이번 프레임에서는 움직임이 있다고 판단

            if not is_currently_in_motion:
                # 이전에는 움직임이 없었는데, 지금 처음 발생한 경우 → 새 구간 시작
                is_currently_in_motion = True
                current_segment_start_index = current_frame_index

            # tail 프레임 카운트 리셋 (나중에 서서히 줄여나감)
            remaining_tail_frames = tail_frames_after_motion

            # 움직임이 있는 프레임은 무조건 저장
            motion_video_writer.write(current_color_frame)

        else:
            # 이번 프레임에서는 변화가 크지 않음 (움직임이 약함 / 없음)
            if is_currently_in_motion:
                if remaining_tail_frames > 0:
                    # tail 프레임 구간: 아직은 구간 안에 포함
                    motion_video_writer.write(current_color_frame)
                    remaining_tail_frames -= 1
                else:
                    # tail도 끝났으므로 구간 종료
                    is_currently_in_motion = False
                    if current_segment_start_index is not None:
                        motion_segments.append(
                            (current_segment_start_index, current_frame_index)
                        )
                    current_segment_start_index = None
            # is_currently_in_motion == False 인 경우는
            # 아무 구간에도 속하지 않으므로 그냥 버린다.

        # 다음 루프에서 사용하기 위해 현재 프레임을 previous로 교체
        previous_gray_frame = current_gray_frame
        current_frame_index += 1

    # 루프가 끝났는데, 여전히 움직임 구간이 열려 있는 경우 마지막 구간 마무리
    if is_currently_in_motion and current_segment_start_index is not None:
        motion_segments.append((current_segment_start_index, current_frame_index - 1))

    # 자원 정리
    video_capture.release()
    motion_video_writer.release()

    return motion_segments


# -----------------------------
# 4. 사람 유무 + 박스 시각화 예시 (실시간 보기용)
# -----------------------------

def show_cctv_with_people_detection(video_source=0):
    """
    CCTV / 웹캠 화면에 사람 검출 결과를 그려서
    실시간으로 보여주는 간단한 데모 함수.

    Parameters
    ----------
    video_source : int or str
        0 → 기본 웹캠
        "cctv.mp4" → 영상 파일
        "rtsp://..." → CCTV 스트림 주소
    """
    video_capture = open_video_source(video_source)

    while True:
        has_frame, color_frame = video_capture.read()
        if not has_frame:
            print("더 이상 읽을 프레임이 없습니다. 종료합니다.")
            break

        # 한 프레임에서 사람 검출
        person_boxes, person_count = detect_people_in_frame(color_frame)

        # 사람 주변에 초록색 박스 그리기
        for (x, y, w, h) in person_boxes:
            cv2.rectangle(color_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 상태 텍스트 만들기
        if person_count > 0:
            status_text = f"Occupied - People: {person_count}"
            status_color = (0, 0, 255)  # 빨간색
        else:
            status_text = "Available - No People"
            status_color = (0, 255, 0)  # 초록색

        # 화면 상단에 상태 박스 + 텍스트 표시
        cv2.rectangle(color_frame, (0, 0), (350, 40), (0, 0, 0), -1)
        cv2.putText(
            color_frame,
            status_text,
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            status_color,
            2,
            cv2.LINE_AA
        )

        cv2.imshow("CCTV People Detection Demo", color_frame)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()


# -----------------------------
# 5. 단독 실행 테스트용
# -----------------------------

if __name__ == "__main__":
    # 1) 실시간 사람 검출 화면 보고 싶을 때 (웹캠 기준)
    # show_cctv_with_people_detection(video_source=0)

    # 2) CCTV 영상에서 움직임 있는 부분만 추출해서 저장하고 싶을 때
    segments = extract_motion_video_from_cctv(
        input_video_source="cctv_sample.mp4",      # 테스트용 CCTV 영상 파일 경로
        output_motion_video_path="motion_only.mp4"
    )
    print("움직임 구간(프레임 단위):", segments)
=======
"""
필요 패키지:
    pip install opencv-python numpy
"""

import cv2
import os
import math
import numpy as np
from pathlib import Path


# --------------------------------------------------
# 기본 유틸함수
# --------------------------------------------------
'''
capture == cv2.VideoCapture 객체
writer == cv2.VideoWriter 객체 
fourcc == cv2.VideoWriter_fourcc 객체
'''
def _open_video(input_video):
    capture = cv2.VideoCapture(str(input_video))
    if not capture.isOpened():
        raise FileNotFoundError(f"영상 파일을 열 수 없습니다: {input_video}")
    return capture


def _make_writer(output_video, fps, width, height, is_color=True): #is_color가 True면 컬러 영상, False면 흑백 영상
    output_video = Path(output_video)
    output_video.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # .mp4 용
    writer = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height), is_color)
    if not writer.isOpened():
        raise RuntimeError(f"VideoWriter 생성 실패: {output_video}")
    return writer


# --------------------------------------------------
# 해상도 & FPS 통일
# --------------------------------------------------

def resize_video(input_video, output_video, width=640, height=360):
    """
    영상 해상도 변경 (fps는 원본 유지)
    """
    capture = _open_video(input_video)
    fps = capture.get(cv2.CAP_PROP_FPS)

    writer = _make_writer(output_video, fps, width, height, is_color=True)

    while True:
        ret, frame = capture.read()
        if not ret:
            break

        resized = cv2.resize(frame, (width, height))
        writer.write(resized)

    capture.release()
    writer.release()
    print(f"[resize_video] 저장 완료: {output_video}")


def change_fps(input_video, output_video, target_fps=15):
    """
    FPS 변경 (간단 버전: 일정 간격으로 프레임 샘플링)
    """
    capture = _open_video(input_video)
    orig_fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if orig_fps == 0:
        raise ValueError("원본 FPS를 읽을 수 없습니다.")

    frame_interval = max(int(round(orig_fps / target_fps)), 1)

    writer = _make_writer(output_video, target_fps, width, height, is_color=True)

    frame_idx = 0
    while True:
        ret, frame = capture.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            writer.write(frame)
        frame_idx += 1

    capture.release()
    writer.release()
    print(f"[change_fps] 저장 완료: {output_video}")


# --------------------------------------------------
# 구간 자르기 / 클립 나누기
# --------------------------------------------------

def trim_video(input_video, output_video, start_sec, end_sec):
    """
    start_sec ~ end_sec 구간만 잘라서 새 영상으로 저장
    """
    capture = _open_video(input_video)
    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)

    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    writer = _make_writer(output_video, fps, width, height, is_color=True)

    current = start_frame
    while current <= end_frame:
        ret, frame = capture.read()
        if not ret:
            break
        writer.write(frame)
        current += 1

    capture.release()
    writer.release()
    print(f"[trim_video] 저장 완료: {output_video}")


def split_video(input_video, segment_sec, output_dir):
    """
    영상 전체를 segment_sec(초) 단위로 연속 잘라서 여러 파일로 저장
    예: segment_sec=5 → 0~5초, 5~10초, ...
    """
    capture = _open_video(input_video)
    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    total_sec = total_frames / fps
    num_segments = math.ceil(total_sec / segment_sec)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    segment_idx = 0
    frame_idx = 0
    writer = None

    while True:
        ret, frame = capture.read()
        if not ret:
            break

        current_sec = frame_idx / fps
        new_segment_idx = int(current_sec // segment_sec)

        if writer is None or new_segment_idx != segment_idx:
            # 새 segment 시작
            if writer is not None:
                writer.release()
            segment_idx = new_segment_idx
            out_path = output_dir / f"segment_{segment_idx:03d}.mp4"
            writer = _make_writer(out_path, fps, width, height)

        writer.write(frame)
        frame_idx += 1

    if writer is not None:
        writer.release()
    capture.release()
    print(f"[split_video] {num_segments}개 세그먼트로 분할 완료: {output_dir}")


# --------------------------------------------------
# 프레임 추출 / 샘플링
# --------------------------------------------------

def extract_frames(input_video, output_dir, every_n_frames=1):
    """
    every_n_frames마다 프레임를 이미지로 저장
    예: every_n_frames=30 → 30프레임마다 1장
    """
    capture = _open_video(input_video)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_idx = 0
    saved = 0

    while True:
        ret, frame = capture.read()
        if not ret:
            break

        if frame_idx % every_n_frames == 0:
            out_path = output_dir / f"frame_{frame_idx:06d}.jpg"
            cv2.imwrite(str(out_path), frame)
            saved += 1

        frame_idx += 1

    capture.release()
    print(f"[extract_frames] {saved}개 프레임 저장 완료: {output_dir}")


def sample_frames(input_video, num_frames=16):
    """
    영상 전체에서 num_frames개 프레임을 균등 간격으로 샘플링해서
    numpy 배열(list)로 반환 (메모리 안에서만 사용)
    """
    capture = _open_video(input_video)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        raise ValueError("영상 길이를 읽을 수 없습니다.")

    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []

    for idx in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = capture.read()
        if not ret:
            continue
        frames.append(frame)

    capture.release()
    frames = np.array(frames)  # shape: (num_frames, H, W, 3)
    print(f"[sample_frames] 샘플링된 프레임 shape: {frames.shape}")
    return frames


# --------------------------------------------------
#좌우반전 (Flip)
# --------------------------------------------------

def horizontal_flip_video(input_video, output_video):
    """
    영상을 좌우 반전한 새 영상 생성 (데이터 증강)
    """
    capture = _open_video(input_video)
    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = _make_writer(output_video, fps, width, height)

    while True:
        ret, frame = capture.read()
        if not ret:
            break

        flipped = cv2.flip(frame, 1)  # 1: 좌우 반전
        writer.write(flipped)

    capture.release()
    writer.release()
    print(f"[horizontal_flip_video] 저장 완료: {output_video}")
# --------------------------------------------------
#밝기 조절 (brightness adjustment)
# --------------------------------------------------

def adjust_brightness_video(input_video, output_video, factor=1.2):
    """
    밝기 조절 (factor > 1 밝게, factor < 1 어둡게)
    """
    capture = _open_video(input_video)
    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = _make_writer(output_video, fps, width, height)

    while True:
        ret, frame = capture.read()
        if not ret:
            break

        img = frame.astype(np.float32)
        img = img * factor
        img = np.clip(img, 0, 255).astype(np.uint8)

        writer.write(img)

    capture.release()
    writer.release()
    print(f"[adjust_brightness_video] 저장 완료: {output_video}")

# --------------------------------------------------
# 영상 회전(rotate)
# --------------------------------------------------
def rotate_video(input_video, output_video, angle=90, scale=1.0):
    """
    영상을 주어진 각도(angle)만큼 회전한 새 영상 생성.
    angle: 시계 반대 방향 회전 (도 단위)
    """
    capture = _open_video(input_video)
    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    center = (width / 2, height / 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)

    writer = _make_writer(output_video, fps, width, height)

    while True:
        ret, frame = capture.read()
        if not ret:
            break
        rotated = cv2.warpAffine(frame, M, (width, height))
        writer.write(rotated)

    capture.release()
    writer.release()
    print(f"[rotate_video] 저장 완료: {output_video}")
#-------------------------------------------------
# 크롭(crop)
#--------------------------------------------------
def crop_video(input_video, output_video, x, y, w, h):
    """
    영상에서 (x, y)을 기준으로 w*h만큼 잘라낸 새 영상 생성
    """
    capture = _open_video(input_video)
    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if x < 0 or y < 0 or w <= 0 or h <= 0:
        raise ValueError("크롭 영역(x, y, w, h)을 확인하세요.")
    if x + w > width or y + h > height:
        raise ValueError("크롭 영역이 원본 영상 범위를 벗어납니다.")

    writer = _make_writer(output_video, fps, w, h)

    while True:
        ret, frame = capture.read()
        if not ret:
            break
        cropped = frame[y:y+h, x:x+w]
        writer.write(cropped)

    capture.release()
    writer.release()
    print(f"[crop_video] 저장 완료: {output_video}")
#------------------------------------------------
#영상자막(text overlay)
#-------------------------------------------------
def add_text_overlay(
    input_video,
    output_video,
    text="Sample Text",
    position=(50, 50),
    font=cv2.FONT_HERSHEY_SIMPLEX,
    font_scale=1.0,
    color=(0, 255, 0),
    thickness=2,
):
    """
    영상 위에 텍스트(자막)를 덮어 새 영상 생성
    """
    capture = _open_video(input_video)
    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = _make_writer(output_video, fps, width, height)

    while True:
        ret, frame = capture.read()
        if not ret:
            break

        img = frame.copy()
        cv2.putText(
            img, text, position,
            font, font_scale, color, thickness,
            lineType=cv2.LINE_AA
        )
        writer.write(img)

    capture.release()
    writer.release()
    print(f"[add_text_overlay] 저장 완료: {output_video}")


# --------------------------------------------------
#영상 합치기 병합기능
# --------------------------------------------------
def concat_videos(video_list, output_video):
    """
    여러 영상을 순서대로 이어붙여 하나의 mp4로 병합
    """
    if not video_list:
        raise ValueError("video_list가 비어 있습니다.")

    # 기준(첫 번째 영상) 정보
    first_cap = _open_video(video_list[0])
    fps = first_cap.get(cv2.CAP_PROP_FPS)
    width = int(first_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(first_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    first_cap.release()

    writer = _make_writer(output_video, fps, width, height)

    for path in video_list:
        capture = _open_video(path)
        while True:
            ret, frame = capture.read()
            if not ret:
                break

            # 해상도 불일치 시 resize
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height))

            writer.write(frame)
        capture.release()

    writer.release()
    print(f"[concat_videos] {len(video_list)}개 영상 병합 완료: {output_video}")

#  --------------------------------------------------
#진행 상황 표시 (progress bar) #pip install tqdm이 필요함
# --------------------------------------------------

#pip install tqdm

try:
    from tqdm import tqdm
    _USE_TQDM = True
except ImportError:
    _USE_TQDM = False


def _progress_iter(capture, total_frames=None, desc="Processing"):
    """
    capture.read() 루프에 tqdm 진행률을 입히는 generator.
    tqdm이 없으면 그냥 (ret, frame)을 yield.
    """
    if total_frames is None or total_frames <= 0:
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    if _USE_TQDM:
        pbar = tqdm(total=total_frames, desc=desc)
    else:
        pbar = None

    while True:
        ret, frame = capture.read()
        if not ret:
            break
        yield ret, frame
        if pbar is not None:
            pbar.update(1)

    if pbar is not None:
        pbar.close()

# --------------------------------------------------
#FPS 변환 고급 버전 (프레임 보간, slow/fast motion)
# --------------------------------------------------

def change_fps(input_video, output_video, target_fps=15):
    """
    FPS 변경 (기본 형태: 원본 프레임을 일정 간격으로 샘플링)
    """
    capture = _open_video(input_video)
    orig_fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if orig_fps == 0:
        raise ValueError("원본 FPS를 읽을 수 없습니다.")
    if target_fps <= 0:
        raise ValueError("target_fps는 0보다 커야 합니다.")

    # 샘플링 간격
    frame_interval = max(int(round(orig_fps / target_fps)), 1)

    writer = _make_writer(output_video, target_fps, width, height, is_color=True)

    frame_idx = 0
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    if _USE_TQDM:
        pbar = tqdm(total=total_frames, desc="Changing FPS")
    else:
        pbar = None

    while True:
        ret, frame = capture.read()
        if not ret:
            break

        # 일정 비율로 프레임만 선택
        if frame_idx % frame_interval == 0:
            writer.write(frame)

        frame_idx += 1
        if pbar is not None:
            pbar.update(1)

    if pbar is not None:
        pbar.close()

    capture.release()
    writer.release()
    print(f"[change_fps] 저장 완료: {output_video}")

# --------------------------------------------------
# 예시 실행용 (직접 돌려볼 때)
# --------------------------------------------------

if __name__ == "__main__":
    src = "input.mp4"  # 테스트용 원본 영상 경로

    # 1. 해상도 줄이기
    resize_video(src, "out_resized.mp4", width=640, height=360)

    # 2. FPS 변경
    change_fps(src, "out_15fps.mp4", target_fps=15)

    # 3. 5~10초 구간만 자르기
    trim_video(src, "out_trim_5_10.mp4", start_sec=5, end_sec=10)

    # 4. 5초 단위로 분할
    split_video(src, segment_sec=5, output_dir="segments")

    # 5. 30프레임마다 이미지 저장
    extract_frames(src, "frames_every30", every_n_frames=30)

    # 6. 영상에서 16프레임 샘플링해서 메모리에 가져오기
    frames = sample_frames(src, num_frames=16)

    # 7. 좌우 반전 영상
    horizontal_flip_video(src, "out_flipped.mp4")

    # 8. 밝게 만든 영상
    adjust_brightness_video(src, "out_bright.mp4", factor=1.3)

    # 9. 영상 회전
    rotate_video(src, "out_rotated.mp4", angle=90)

    #10. 영상 크롭
    crop_video(src, "out_cropped.mp4", x=100, y=50 , w=400, h=300)

    #11. 영상에 자막 넣기
    add_text_overlay(
        src,
        "out_text_overlay.mp4",
        text="Hello, World!",
        position=(100, 100),
        font_scale=2.0,
        color=(255, 0, 0),
        thickness=3,
    )

    #12. 영상 병합
    concat_videos(
        [src, "out_resized.mp4", "out_15fps.mp4"],
        "out_concatenated.mp4"
    )

    #13. FPS 변경 (진행 상황 표시)
    change_fps(src, "out_10fps_progress.mp4", target_fps=10)

# --------------------------------------------------
>>>>>>> 19e641515f6d374335015613f761eee663354656
