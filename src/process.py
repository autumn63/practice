import librosa
import numpy as np

def wav_del_space(y, sr):
    print(f"--- Audio Analysis Start (Total Samples: {len(y)}) ---")

    # 1. 예외 처리: 데이터가 비어있으면 종료
    if len(y) == 0: return []

    # 2. 정규화 (필수): 소리 크기를 0~1 사이로 맞춤
    y = librosa.util.normalize(y)

    # 3. 공백 제거 (top_db=20 고정)
    # 20dB: 잡음과 말소리를 구분하는 기준. 낮을수록 엄격하게 자름.
    intervals = librosa.effects.split(
        y, 
        top_db=20,          # 요청하신 고정값
        frame_length=1024, 
        hop_length=256
    )

    segments_list = []

    # 4. 구간 필터링 및 저장
    for start, end in intervals:
        # 0.1초(노이즈)보다 긴 구간만 유효한 데이터로 인정
        if (end - start) > sr * 0.1:
            segments_list.append(y[start:end])

    return segments_list