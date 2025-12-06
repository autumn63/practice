import os
import numpy as np
import file, process

def main():
    # 경로 및 파일 설정
    base_dir = "data"
    input_dir = os.path.join(base_dir, "input")
    output_dir = os.path.join(base_dir, "output")
    split_dir = os.path.join(output_dir, "split_files")
    
    target_filename = "video4.wav"  # input 폴더에 있는 wav 파일명
    input_path = os.path.join(input_dir, target_filename)

    # 오디오 파일 로드
    try:
        y, sr = file.load(input_path)
    except Exception as e:
        print(f"Error: 파일을 불러올 수 없습니다. 경로를 확인하세요.\n{e}")
        return

    # 파형 시각화 및 이미지 저장
    wav_graph = process.show_wav(y, sr)
    wav_graph.savefig(os.path.join(output_dir, "waveform.png"))
    print("hi")
    wav_graph.close() # 메모리 해제

    # 무음 제거 및 구간 분할
    segments = process.wav_del_space(y, sr)

    # 결과물 저장
    if segments:
        # 분할된 파일 개별 저장
        file.save(split_dir, segments, sr)

        # 공백 제거된 합본 생성 및 저장
        merged_y = np.concatenate(segments)
        merged_path = os.path.join(output_dir, "merged_no_silence.wav")
        file.save_merged(merged_path, merged_y, sr)
        
        print(f"{len(segments)}개 저장.")
    else:
        print("Error.. 저장실패")

if __name__ == "__main__":
    main()