# =================================================================
# 1. 라이브러리 및 설정
# =================================================================
import cv2            # 동영상 및 이미지 처리를 위한 OpenCV 라이브러리
import numpy as np    # 배열 처리를 위한 Numpy 라이브러리
import os             # 파일 시스템 관리를 위한 OS 라이브러리
import glob           # 파일 목록 검색을 위한 Glob 라이브러리

# --- 사용자 설정 영역 ---
INPUT_VIDEO_PATH = 'your_video.mp4'         # ⭐ 처리할 원본 동영상 파일 경로
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