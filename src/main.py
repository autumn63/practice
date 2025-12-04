# main.py
import os
import sys

# src 폴더 안에 있는 파일들을 불러옴
import file, process, convert

# 메인 실행 함수 
def main():
    print("Start Processing")

    # 설정값 정의(일단 video1 파일 소스 사용)
    video_filename = "video2.mp4"  # 파일 이름만 깔끔하게 정의
    output_folder = os.path.join("data", "output") # 경로 안전하게 생성

    # 비디오 to wav 변환
    # convert함수가 'data/output/video2.mp3' 전체 경로를 리턴
    audio_file_path = convert.convert(video_filename)

    # 변환 실패 시 중단 (안전장치)
    if audio_file_path == -1:
        print("Conver Error Occurred")
        return

    # 파일 불러오기
    y, sr = file.load(audio_file_path)

    # 공백 기준으로 오디오 편집
    segments = process.wav_del_space(y, sr)
    print(f"Number of segments: {len(segments)}")

    # 편집된 오디오 파일 저장
    # os.path.join을 사용하여 운영체제 상관없이 경로 생성
    save_path = os.path.join(output_folder, "edited_audio")
    file.save(save_path, segments, sr)

if __name__ == "__main__":
    main()