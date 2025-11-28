#라이브러리 설치
import cv2        # 동영상 및 이미지 처리
import numpy as np
import os

# 프레임 추출 (동영상 -> 이미지 시퀀스)
video_path = 'your_video.mp4'           #동영상 불러오기
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():                  #열리지 않을 경우
    print(f"Error: Could not open video file {video_path}")
    exit()

#영상 기본 정보 확인
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # 전체 프레임 수
fps = cap.get(cv2.CAP_PROP_FPS)                      # 초당 프레임 수
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))       # 프레임 너비
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))     # 프레임 높이

print(f"Total Frames: {frame_count}, FPS: {fps}, Resolution: {width}x{height}") #영상 정보 출력
