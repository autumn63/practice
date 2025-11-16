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
# 기본 유틸
# --------------------------------------------------

def _open_video(input_path):
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"영상 파일을 열 수 없습니다: {input_path}")
    return cap


def _make_writer(output_path, fps, width, height, is_color=True):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # .mp4 용
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height), is_color)
    if not writer.isOpened():
        raise RuntimeError(f"VideoWriter 생성 실패: {output_path}")
    return writer


# --------------------------------------------------
# 1. 해상도 & FPS 통일
# --------------------------------------------------

def resize_video(input_path, output_path, width=640, height=360):
    """
    영상 해상도 변경 (fps는 원본 유지)
    """
    cap = _open_video(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    writer = _make_writer(output_path, fps, width, height, is_color=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        resized = cv2.resize(frame, (width, height))
        writer.write(resized)

    cap.release()
    writer.release()
    print(f"[resize_video] 저장 완료: {output_path}")


def change_fps(input_path, output_path, target_fps=15):
    """
    FPS 변경 (간단 버전: 일정 간격으로 프레임 샘플링)
    """
    cap = _open_video(input_path)
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if orig_fps == 0:
        raise ValueError("원본 FPS를 읽을 수 없습니다.")

    frame_interval = max(int(round(orig_fps / target_fps)), 1)

    writer = _make_writer(output_path, target_fps, width, height, is_color=True)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"[change_fps] 저장 완료: {output_path}")


# --------------------------------------------------
# 2. 구간 자르기 / 클립 나누기
# --------------------------------------------------

def trim_video(input_path, output_path, start_sec, end_sec):
    """
    start_sec ~ end_sec 구간만 잘라서 새 영상으로 저장
    """
    cap = _open_video(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    writer = _make_writer(output_path, fps, width, height, is_color=True)

    current = start_frame
    while current <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)
        current += 1

    cap.release()
    writer.release()
    print(f"[trim_video] 저장 완료: {output_path}")


def split_video(input_path, segment_sec, output_dir):
    """
    영상 전체를 segment_sec(초) 단위로 연속 잘라서 여러 파일로 저장
    예: segment_sec=5 → 0~5초, 5~10초, ...
    """
    cap = _open_video(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    total_sec = total_frames / fps
    num_segments = math.ceil(total_sec / segment_sec)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    segment_idx = 0
    frame_idx = 0
    writer = None

    while True:
        ret, frame = cap.read()
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
    cap.release()
    print(f"[split_video] {num_segments}개 세그먼트로 분할 완료: {output_dir}")


# --------------------------------------------------
# 3. 프레임 추출 / 샘플링
# --------------------------------------------------

def extract_frames(input_path, output_dir, every_n_frames=1):
    """
    every_n_frames마다 프레임를 이미지로 저장
    예: every_n_frames=30 → 30프레임마다 1장
    """
    cap = _open_video(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_idx = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % every_n_frames == 0:
            out_path = output_dir / f"frame_{frame_idx:06d}.jpg"
            cv2.imwrite(str(out_path), frame)
            saved += 1

        frame_idx += 1

    cap.release()
    print(f"[extract_frames] {saved}개 프레임 저장 완료: {output_dir}")


def sample_frames(input_path, num_frames=16):
    """
    영상 전체에서 num_frames개 프레임을 균등 간격으로 샘플링해서
    numpy 배열(list)로 반환 (메모리 안에서만 사용)
    """
    cap = _open_video(input_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        raise ValueError("영상 길이를 읽을 수 없습니다.")

    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frames.append(frame)

    cap.release()
    frames = np.array(frames)  # shape: (num_frames, H, W, 3)
    print(f"[sample_frames] 샘플링된 프레임 shape: {frames.shape}")
    return frames


# --------------------------------------------------
# 4. 간단한 데이터 증강 (Flip / 밝기 조절)
# --------------------------------------------------

def horizontal_flip_video(input_path, output_path):
    """
    영상을 좌우 반전한 새 영상 생성 (데이터 증강)
    """
    cap = _open_video(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = _make_writer(output_path, fps, width, height)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        flipped = cv2.flip(frame, 1)  # 1: 좌우 반전
        writer.write(flipped)

    cap.release()
    writer.release()
    print(f"[horizontal_flip_video] 저장 완료: {output_path}")


def adjust_brightness_video(input_path, output_path, factor=1.2):
    """
    밝기 조절 (factor > 1 밝게, factor < 1 어둡게)
    """
    cap = _open_video(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = _make_writer(output_path, fps, width, height)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img = frame.astype(np.float32)
        img = img * factor
        img = np.clip(img, 0, 255).astype(np.uint8)

        writer.write(img)

    cap.release()
    writer.release()
    print(f"[adjust_brightness_video] 저장 완료: {output_path}")


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
