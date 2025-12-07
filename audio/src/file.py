import soundfile as sf
import librosa
import os

def save(base_path, segments, sr):
    # 여러 오디오 조각을 개별 파일로 저장
    if not os.path.exists(base_path):
        os.makedirs(base_path, exist_ok=True)

    if not segments:
        return

    for i, segment in enumerate(segments):
        file_name = f"segment_{i+1:03d}.wav"
        full_path = os.path.join(base_path, file_name)
        sf.write(full_path, segment, sr)

def save_merged(file_path, data, sr):
    # 합쳐진 오디오 데이터를 하나의 파일로 저장
    folder = os.path.dirname(file_path)
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
        
    sf.write(file_path, data, sr)

def load(input_path): 
    # 오디오 파일 읽기
    y, sr = librosa.load(input_path)
    return y, sr