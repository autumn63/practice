#라이브러리 설치
import numpy as np
import librosa, librosa.display #!pip install --ignore-installed librosa 로 librosa 설치필요
import matplotlib.pyplot as plt
import soundfile as sf


#무엇을 위한 전처리? -> 일단 지금은 시각화 + 공백 제거 + 특징 추출정도로만 진행해봄
#-------------------------------------------------------


# wav 파일 불러오기
file = "src/example.wav"

# Librosa로 오디오 불러오기
y, sr = librosa.load(file) # y : 파형의 amplitude값, sr : sampling rate

# Draw WavePlot
def draw_waveplot(y, sr):
    plt.figure(figsize=(15, 10)) #
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


# wav 파일 Edit(공백제거)
# 우선 절반으로 나눠보기
def wav_del_space(y, sr):
    half = len(y) // 2
    y2 = y[:half]  # 절반만 사용
    time2 = np.linspace(0, len(y2) / sr, num=len(y2))

    return y2, time2

    # 시각화
    """
    plt.figure(figsize=(15, 5))
    plt.plot(time2, y2)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Waveform (First Half)")
    plt.savefig("src/edited_example.png")  # 편집된 파일 저장
    plt.show()
    """


#함수 실행해보기.. !!!!!!나중에 지워야함
y2, time2 = wav_del_space(y, sr)

#자른 음성 저장. #librosa에서는 output기능이 사라짐;; 대신 soundfile 라이브러리 사용할 예정.
sf.write("src/edited_example.wav", y2, sr)
