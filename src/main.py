import os
import numpy as np  # [필수] 배열 합치기 위해 필요
import file, process, convert

def main():
    # 1. 파일 설정
    video_filename = "video2.mp4" 
    output_folder = os.path.join("data", "output")
    
    # 결과물 폴더 분리 (깔끔하게)
    split_output_folder = os.path.join(output_folder, "split_files")

    # 2. 변환 (mp4 -> wav)
    #print("Converting video to audio...")
    #audio_path = convert.convert(video_filename)
    
    audio_path = "data/input/example.wav"

    # 3. 로드
    try:
        y, sr = file.load(audio_path)
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    # 4. 처리 (공백 제거 및 분할)
    segments = process.wav_del_space(y, sr)

    # 5. 저장 단계
    if segments:
        # (A) 분할된 파일들 저장
        file.save(split_output_folder, segments, sr)
        print(f"Done. {len(segments)} split files saved.")

        # (B) [핵심] 공백 제거된 합본 만들기
        print("Mering segments...")
        merged_y = np.concatenate(segments) # 리스트에 있는 조각들을 일렬로 이어 붙임
        
        # 합본 파일 경로 설정
        merged_filename = f"merged_no_silence.wav"
        merged_path = os.path.join(output_folder, merged_filename)
        
        # 저장
        file.save_merged(merged_path, merged_y, sr)

    else:
        print("Warning: 저장할 오디오 구간이 없습니다.")

if __name__ == "__main__":
    main()