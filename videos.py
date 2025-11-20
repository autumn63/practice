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
