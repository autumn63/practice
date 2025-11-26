#라이브러리 설치
import numpy as np
import librosa, librosa.display
import matplotlib.pyplot as plt

#오디오 파일 불러오기
#리샘플링
#모도변환
#정규화

#패딩 및 자르기

#특징 추출
# ....등등 순차적으로 진행하기 

#-------------------------------------------------------
# wav 파일 불러오기
file = "src/example.wav"

# Librosa로 오디오 불러오기
y, sr = librosa.load(file) # y : 파형의 amplitude값, sr : sampling rate

plt.figure(figsize=(15, 10)) # Wavform 시각화
librosa.display.waveshow(y, sr=sr, alpha=0.5)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Waveform")

plt.show()