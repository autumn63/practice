import soundfile as sf

# Sav edit FIlE
def save(filename, segment, sr):
    
    for i, segment in enumerate(segment):
        save_name = f"{filename}_{i}.wav"

        sf.write(save_name, segment, sr)  # 편집된 오디오 파일 저장
        print(f"Saved: {save_name}")

def load(input_path):
    # 오디오 파일 불러오기
    y, sr = sf.read(input_path)
    return y, sr