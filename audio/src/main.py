import os
import numpy as np
import file, process, convert

def main():
    # 경로 설정
    base_dir = "audio/data"
    output_dir = os.path.join(base_dir, "output")
    split_dir = os.path.join(output_dir, "split_files")
    
    # 처리할 비디오 파일명 (audio/data/input 폴더 안에 있어야 함)
    target_video = "video4.mp4" 

    # 1. 변환 (MP4 -> WAV)
    audio_path = convert.convert(base_dir, target_video)
    
    if not audio_path:
        return # 변환 실패 시 종료

    # 2. 로드
    try:
        y, sr = file.load(audio_path)
    except Exception as e:
        print(f"Error... 파일 오류")
        return

    # 3. 파형 시각화
    wav_graph = process.show_wav(y, sr)
    wav_graph.savefig(os.path.join(output_dir, "waveform.png"))
    wav_graph.close()

    # 4. 무음 제거 및 구간 분할
    segments = process.wav_del_space(y, sr)

    # 5. 저장
    if segments:
        # 분할 파일 저장
        file.save(split_dir, segments, sr)

        # 합본 파일 저장
        merged_y = np.concatenate(segments)
        merged_path = os.path.join(output_dir, "merged_no_silence.wav")
        file.save_merged(merged_path, merged_y, sr)
        
        print(f"{len(segments)}개 저장")
    else:
        print("Error... 오디오 구간 오류")

if __name__ == "__main__":
    main()