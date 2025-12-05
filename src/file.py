import soundfile as sf
import os

def save(base_path, segments, sr):
    """여러 조각을 개별 파일로 저장"""
    if not os.path.exists(base_path):
        os.makedirs(base_path, exist_ok=True)

    if not segments:
        return

    for i, segment in enumerate(segments):
        file_name = f"segment_{i+1:03d}.wav"
        full_path = os.path.join(base_path, file_name)
        sf.write(full_path, segment, sr)

def save_merged(file_path, data, sr):
    """[추가됨] 합쳐진 데이터 하나를 저장"""
    # 저장할 폴더가 없으면 생성
    folder = os.path.dirname(file_path)
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
        
    sf.write(file_path, data, sr)

def load(input_path): 
    y, sr = sf.read(input_path)
    return y, sr