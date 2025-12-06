import librosa
import numpy as np

def wav_del_space(y, sr):
    print(f"--- Audio Analysis Start (Total Samples: {len(y)}) ---")

    # 데이터가 비어있으면 빈 리스트 반환하기
    if len(y) == 0: return []

    # 정규화 --> 소리 크기를 0~1 사이로 맞춤
    y = librosa.util.normalize(y)

    # 공백 제거 (top_db=20)이 그나마 정확한듯
    intervals = librosa.effects.split(
        y, 
        top_db=20,         
        frame_length=1024, 
        hop_length=256
    )

    segments_list = []

    # 구간 필터링 및 저장
    for start, end in intervals:
        # 0.1초(노이즈)보다 긴 구간만 저장하고 나머지는 버리기
        if (end - start) > sr * 0.1:
            segments_list.append(y[start:end])

    return segments_list