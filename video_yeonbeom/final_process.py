# =================================================================
# 1. 라이브러리 및 설정
# =================================================================

import cv2
import glob
import os

# --- 사용자 설정 영역 ---
INPUT_DIR = 'processed_frames_bw_flipped'      # 전처리된 이미지가 저장된 폴더
OUTPUT_VIDEO_NAME = 'processed_video_output.mp4' # ⭐ 출력할 파일을 MP4로 변경 ⭐
TARGET_FPS = 15.0                         # 출력 영상의 FPS (프레임 레이트)
# -------------------------

# =================================================================
# 2. 비디오 라이터(VideoWriter) 설정
# =================================================================

frame_files = sorted(glob.glob(os.path.join(INPUT_DIR, '*.jpg')))

if not frame_files:
    print(f"Error: 폴더 '{INPUT_DIR}'에 저장된 이미지 파일이 없습니다.")
    exit()

first_frame = cv2.imread(frame_files[0], cv2.IMREAD_GRAYSCALE)
height, width = first_frame.shape[:2]

# 코덱 설정: MP4V 코덱 사용 (*'mp4v' 대신 *'XVID'를 .mp4 파일에 사용해도 호환될 때가 많습니다.)
# Mac 환경에서는 'mp4v'를 추천하며, 오류가 발생하면 'XVID'나 'DIVX'를 시도해 보세요.
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 

# VideoWriter 초기화
out = cv2.VideoWriter(OUTPUT_VIDEO_NAME, fourcc, TARGET_FPS, (width, height), isColor=True)

print(f"--- {len(frame_files)}개의 프레임을 MP4 영상으로 변환 시작 ---")
print(f"출력 해상도: {width}x{height}, FPS: {TARGET_FPS}")

# =================================================================
# 3. 프레임 읽고 영상으로 쓰기
# =================================================================

for i, file_path in enumerate(frame_files):
    frame = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    
    # 흑백(1채널) 이미지를 3채널(컬러 형식)로 변환
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    
    out.write(frame_rgb)
    
    if (i + 1) % 100 == 0:
        print(f"{i + 1}번째 프레임 처리 완료...")

# 4. 마무리
out.release()
print("--- 영상 변환 완료 ---")
print(f"최종 MP4 영상 파일이 '{OUTPUT_VIDEO_NAME}'으로 저장되었습니다.")