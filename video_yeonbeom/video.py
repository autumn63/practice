# =================================================================
# 1. 라이브러리 및 설정
# =================================================================

# 필요한 라이브러리 설치 (터미널/명령 프롬프트에 입력해주세요)
# pip install opencv-python
# pip install numpy

import cv2            # 동영상 및 이미지 처리를 위한 OpenCV 라이브러리
import numpy as np    # 배열 처리를 위한 Numpy 라이브러리
import os             # 파일 시스템 관리를 위한 OS 라이브러리
import glob           # 파일 목록 검색을 위한 Glob 라이브러리

# --- 사용자 설정 영역 ---
INPUT_VIDEO_PATH = 'video_yeonbeom.mp4'         # ⭐ 처리할 원본 동영상 파일 경로
PROCESSED_DIR = 'processed_frames_bw_flipped' # 전처리된 프레임을 저장할 디렉토리명
SEQUENCE_LENGTH = 30                        # 모델 입력으로 사용할 연속된 프레임의 개수
TARGET_SIZE = (224, 224)                    # 전처리 후 프레임 크기 (높이, 너비)
FRAME_INTERVAL = 5                          # 5 프레임마다 하나씩 추출 (샘플링)
# -------------------------

# =================================================================
# 2. 프레임 추출, 전처리 (흑백, 좌우반전) 및 저장
# =================================================================

# 2-1. 동영상 파일 열기
cap = cv2.VideoCapture(INPUT_VIDEO_PATH)

if not cap.isOpened():
    print(f"Error: Could not open video file {INPUT_VIDEO_PATH}")
    # 파일이 없거나 경로가 잘못되었을 경우 프로그램 종료
    exit()

# 2-2. 영상 기본 정보 확인 및 출력
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # 전체 프레임 수
fps = cap.get(cv2.CAP_PROP_FPS)                      # 초당 프레임 수
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))       # 프레임 너비
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))     # 프레임 높이

print(f"--- 원본 영상 정보 ---")
print(f"Total Frames: {frame_count}, FPS: {fps}, Resolution: {width}x{height}")

# 2-3. 출력 디렉토리 생성
os.makedirs(PROCESSED_DIR, exist_ok=True)
print(f"전처리된 프레임은 '{PROCESSED_DIR}'에 저장됩니다.")

frame_num = 0
saved_count = 0

print("--- 1단계: 프레임 추출 및 전처리 시작 ---")
while cap.isOpened():
    ret, frame = cap.read() # 프레임 읽기

    if not ret:
        break # 동영상 끝에 도달하면 루프 종료

    # 지정된 간격으로 프레임 샘플링 (시간적 데이터 축소)
    if frame_num % FRAME_INTERVAL == 0:
        # 1. 크기 조정 (Resizing): 모든 프레임을 동일한 크기로 맞춤
        resized_frame = cv2.resize(frame, TARGET_SIZE, interpolation=cv2.INTER_AREA)

        # 2. 흑백 변환 (Grayscale): 채널 수를 3(컬러)에서 1(흑백)로 줄임
        gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY) 

        # 3. 좌우 반전 (Horizontal Flip): 데이터 증강 또는 특정 방향성 보정을 위해 사용
        # flipCode=1은 좌우 반전을 의미
        flipped_frame = cv2.flip(gray_frame, 1)

        # 4. 0-1 정규화 (Normalization)
        # 모델 학습을 위해 픽셀 값(0-255)을 0.0-1.0 범위로 변환
        normalized_frame = flipped_frame.astype(np.float32) / 255.0

        # --- 저장용: 0-255 범위로 다시 변환 (imwrite는 정수형 0-255 값을 요구) ---
        frame_to_save = (normalized_frame * 255).astype(np.uint8)

        # 5. 프레임 파일로 저장
        frame_filename = os.path.join(PROCESSED_DIR, f'frame_{frame_num:06d}.jpg')
        cv2.imwrite(frame_filename, frame_to_save)
        
        saved_count += 1

    frame_num += 1

cap.release()
print(f"1단계 완료: 총 {frame_num} 프레임 중 {saved_count}개 프레임 저장됨.")