# =================================================================
# 1. 라이브러리 설치 및 설정
# =================================================================

# 필요한 라이브러리 설치 (터미널에 입력해주세요)
# pip install opencv-python numpy

import cv2            # 동영상 및 이미지 처리를 위한 OpenCV
import numpy as np    # 배열 처리를 위한 Numpy
import os             # 파일 시스템 관리를 위한 OS

# --- 내 설정 영역 ---
INPUT_VIDEO_PATH = 'video_yeonbeom.mp4'         # 내가 처리할 동영상 파일 경로
OUTPUT_DIR = 'frames_grayscale_only'            # 흑백 변환한 프레임을 저장할 폴더명

# =================================================================
# 2. 흑백 변환 및 프레임 저장
# =================================================================

# 2-1. 동영상 파일 열기
cap = cv2.VideoCapture(INPUT_VIDEO_PATH)

if not cap.isOpened():
    print(f"Error: 동영상 파일을 열 수 없어: {INPUT_VIDEO_PATH}")
    exit()

os.makedirs(OUTPUT_DIR, exist_ok=True)
frame_num = 0

print(f"--- {OUTPUT_DIR} 처리 시작 (흑백 변환) ---")
while cap.isOpened():
    ret, frame = cap.read() # 프레임 읽기
    if not ret: break       # 끝까지 다 읽으면 종료

    # 1. 흑백 변환 (Grayscale)
    # BGR(컬러) 채널을 GRAY(흑백) 채널로 변환 (채널 수 3 -> 1)
    processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 

    # 2. 정규화 (Normalization)
    # 픽셀 값을 0.0-1.0 범위로 변환
    normalized_frame = processed_frame.astype(np.float32) / 255.0

    # --- 저장용: 다시 0-255 범위로 변환 ---
    frame_to_save = (normalized_frame * 255).astype(np.uint8)

    # 3. 파일로 저장
    frame_filename = os.path.join(OUTPUT_DIR, f'frame_{frame_num:06d}.jpg')
    cv2.imwrite(frame_filename, frame_to_save)
    
    frame_num += 1

cap.release()
print(f"처리 완료. 총 {frame_num}개 프레임이 '{OUTPUT_DIR}'에 저장됨.")