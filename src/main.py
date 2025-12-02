#main.py

# src 폴더 안에 있는 파일들을 불러옴
from . import file, process

# 메인 실행 함수 
def main():
    print("Start Processing")

    # 설정값 정의
    input_path = "data/example.wav"
    output_folder = "results/"

    # 파일 불러오기
    y, sr = file.load(input_path)

    # 공백 기준으로 오디오 편집
    segments = process.wav_del_space(y, sr)
    print(f"Number of segments: {len(segments)}")

    # 편집된 오디오 파일 저장
    file.save(output_folder + "edited_audio", segments, sr)


if __name__ == "__main__":
    main()