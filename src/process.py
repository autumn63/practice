import librosa
import numpy as np

def wav_del_space(y, sr):
    """
    오디오 데이터(y)와 샘플링레이트(sr)를 받아서
    무음 구간을 제거한 '의미 있는 소리 조각들의 리스트'를 반환.
    """
    
    # 반환할 리스트 초기화
    segments_list = []
    y = librosa.util.normalize(y)

    # 무음 구간 기준으로 나누기
    # top_db=25: 잡음이 좀 섞여 있어 기준 데시벨 25로 잡음
    # frame_length, hop_length: 값을 넣어야 너무 자잘하게 끊기는 걸 방지함
    intervals = librosa.effects.split(y, top_db=25, frame_length=2048, hop_length=512)
    
    # 구간별로 잘라서 리스트에 담기
    for start, end in intervals:
        
        # 현재 구간의 길이(end - start)가 0.3초보다 짧으면 무시 (잡음 제거)
        if (end - start) < sr * 0.1:
            continue
        
        # 실제 데이터 자르기
        chunk = y[start:end]
        
        # 잘라낸 조각이 너무 짧으면 (0.1초 미만) 무시
        if len(chunk) < 2048: 
            continue
            
        # 살아남은 조각을 리스트에 추가
        segments_list.append(chunk)

    # 결과 반환 (main.py가 이걸 받아서 저장함)
    return segments_list