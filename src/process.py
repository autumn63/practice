import librosa
import librosa.display
import matplotlib.pyplot as plt

def wav_del_space(y, sr):
    # 데이터 유효성 검사
    if len(y) == 0: 
        return []

    # 정규화
    y = librosa.util.normalize(y)

    # 무음 구간 감지 (top_db 조절로 민감도 설정 가능)
    intervals = librosa.effects.split(
        y, 
        top_db=20,         
        frame_length=1024, 
        hop_length=256
    )
 
    segments_list = []

    # 0.1초 미만의 노이즈 구간 제외하고 리스트에 추가
    for start, end in intervals:
        if (end - start) > sr * 0.1:
            segments_list.append(y[start:end])

    return segments_list

def show_wav(y, sr):
    plt.figure(figsize=(15, 10))
    librosa.display.waveshow(y, sr=sr)
    plt.title("Waveform")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    
    # plt 객체를 반환하여 main에서 저장하도록 함
    return plt