# =================================================================
# 1. 라이브러리 설치 및 설정
# =================================================================

# 필요한 라이브러리 설치
# pip install opencv-python numpy
# cd data_video
import cv2
import numpy as np
import os
import glob
import math # math.gcd 함수를 사용하여 비율을 계산할 때 사용 가능 (선택 사항)

# --- 사용자 설정 영역 ---
INPUT_VIDEO_PATH = 'video/data_video/video.mp4'         # 처리할 원본 동영상 파일 경로
PROCESSED_DIR = 'standardized_frames_16x9'    # 표준화된 프레임을 저장할 디렉토리명
TARGET_ASPECT_RATIO = (16, 9)                   # 목표 비율
TARGET_SIZE = (320, 180)                        # 16:9 비율을 유지하는 크기 (320 / 16 = 20, 180 / 9 = 20)
SEQUENCE_LENGTH = 30                        # 모델 입력으로 사용할 연속된 프레임의 개수
FRAME_INTERVAL = 5                          # 5 프레임마다 하나씩 추출 (샘플링)
# --- 표준화 설정 ---
CLIP_LIMIT = 2.0                            # CLAHE 대비 제한 값 
TILE_GRID_SIZE = (8, 8)                     # CLAHE 처리 영역 크기
# -------------------------


# =================================================================
# 2. 영상 표준화, 프레임 추출, 저장
# =================================================================

def standardize_and_extract_frames(video_path, output_dir, target_size, interval, clip_limit, tile_grid_size):
    """
    동영상을 읽어 해상도(16:9), 밝기/대비, 색감 표준화하여 프레임을 저장
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return 0

    os.makedirs(output_dir, exist_ok=True)
    frame_num = 0
    saved_count = 0
    
    # CLAHE 객체 생성 (밝기/대비 균일화 도구)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    print("--- 1단계: 영상 표준화 및 프레임 추출 시작 (16:9 비율 적용) ---")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        if frame_num % interval == 0:
            
            # 1. 해상도 맞추기 (크기 조정 Resizing)
            # 프레임을 TARGET_SIZE (320x180)으로 조정합니다.
            resized_frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)

            # 2. 밝기 및 대비 균일화 (CLAHE 적용)
            # 2-1. BGR->LAB 변환 (밝기(L) 채널 분리)
            lab = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # 2-2. L 채널에 CLAHE 적용
            cl = clahe.apply(l)
            
            # 2-3. L 채널을 다시 합치고 LAB->BGR로 복원
            limg = cv2.merge((cl, a, b))
            contrast_enhanced_frame = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

            # 3. 색감 정규화 (Normalization)
            # 모든 픽셀 값을 0.0-1.0 범위로 변환합니다.
            normalized_frame = contrast_enhanced_frame.astype(np.float32) / 255.0

            # --- 저장용: 0-255 범위로 다시 변환 ---
            frame_to_save = (normalized_frame * 255).astype(np.uint8)

            # 4. 프레임 파일로 저장
            frame_filename = os.path.join(output_dir, f'frame_{frame_num:06d}.jpg')
            cv2.imwrite(frame_filename, frame_to_save)
            
            saved_count += 1
            
        frame_num += 1

    cap.release()
    print(f"1단계 완료: 총 {frame_num} 프레임 중 {saved_count}개 표준화된 프레임이 저장됨.")
    return saved_count


# =================================================================
# 3. 저장된 프레임들을 시퀀스 데이터셋으로 구성 및 저장
# =================================================================

def create_sequences(frame_directory, sequence_length):
    """
    저장된 개별 프레임들을 불러와 시퀀스 배열(Numpy)로 구성
    """
    frame_files = sorted(glob.glob(os.path.join(frame_directory, '*.jpg')))
    
    if len(frame_files) < sequence_length:
        print("경고: 시퀀스 구성에 필요한 프레임 수가 부족합니다.")
        return np.array([])

    all_frames = []
    
    # 3-1. 저장된 모든 프레임 불러오기
    for file_path in frame_files:
        # BGR(컬러) 이미지로 읽음
        frame = cv2.imread(file_path, cv2.IMREAD_COLOR) 
        
        # 0-1 범위로 재변환 (저장된 파일이 0-255였기 때문에)
        frame = frame.astype(np.float32) / 255.0
        
        all_frames.append(frame)

    all_frames = np.array(all_frames)

    # 3-2. 시퀀스 생성 (오버랩 없음)
    sequences = []
    num_sequences = len(all_frames) // sequence_length
    
    for i in range(num_sequences):
        start_idx = i * sequence_length
        end_idx = start_idx + sequence_length
        
        sequence = all_frames[start_idx:end_idx]
        sequences.append(sequence)

    return np.array(sequences)


# =================================================================
# 4. 전체 실행 로직
# =================================================================

# 1단계 실행
saved_frames_count = standardize_and_extract_frames(
    INPUT_VIDEO_PATH,
    PROCESSED_DIR,
    TARGET_SIZE,
    FRAME_INTERVAL,
    CLIP_LIMIT,
    TILE_GRID_SIZE
)

if saved_frames_count > 0:
    print("--- 2단계: 시퀀스 데이터셋 구성 및 저장 시작 ---")

    # 2단계 실행
    video_dataset = create_sequences(
        frame_directory=PROCESSED_DIR,
        sequence_length=SEQUENCE_LENGTH
    )

    # 결과 확인 및 저장
    if video_dataset.size > 0:
        print("\n--- 💾 최종 데이터셋 구성 완료 ---")
        # 최종 배열 형태: (시퀀스 개수, 시퀀스 길이, 높이, 너비, 채널)
        print(f"데이터셋 형태 (Shape): {video_dataset.shape}")
        
        output_filename = 'video_dataset_16x9.npy'
        np.save(output_filename, video_dataset)
        print(f"최종 데이터셋이 '{output_filename}' 파일로 저장되었습니다.")