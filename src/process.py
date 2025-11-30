import matplotlib.pyplot as plt
import librosa, librosa.display


# Draw WavePlot
def draw_waveplot(y, sr):
    plt.figure(figsize=(15, 10)) 
    librosa.display.waveshow(y, sr=sr, alpha=0.5)

    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Waveform")
    plt.savefig("src/waveplot.png")  # 웨이브플롯 이미지 저장
    plt.show()

# Draw Spectrogram
def draw_spectrogram(y, sr):
    X = librosa.stft(y)  # 데이타의 스펙트로그램 리턴 
    # shift : 음성을 시간기반 to 주파수 기반으로 변환.

    db = librosa.amplitude_to_db(abs(X))  # 스펙트로그램을 데시벨로 변환

    plt.figure(figsize=(15, 10))
    librosa.display.specshow(db, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.savefig("src/spectrogram.png")  # 스펙트로그램 이미지 저장
    plt.show()


# wav 파일 Edit(공백제거 --> 무음 구간 기준으로 나누기)
def wav_del_space(y, sr):

    segment = []
    intervals = librosa.effects.split(y, top_db=30) # intervals : 무음이 아닌 구간의 시작과 끝 인덱스 반환

    for i, (start, end) in enumerate(intervals):
        # 간격이 너무 짧은 구간은 무시
        if len(segment) < sr * 0.5:
            continue

        segment.append(y[start:end]) # 무음이 아닌 구간을 segment 리스트에 추가

    return segment