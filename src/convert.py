import os
from moviepy import VideoFileClip

def convert(filename):
    # 경로 설정
    base_dir = "data"  # 영상이 있는 폴더
    output_dir = os.path.join(base_dir, "output")
    
    # 폴더가 없으면 생성
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    video_path = os.path.join(base_dir + '/input', filename)
    audio_filename = os.path.splitext(filename)[0] + ".wav"
    output_path = os.path.join(output_dir, audio_filename)

    print(f"Converting: {video_path} -> {output_path}")

    # 변환 작업
    try:
        video = VideoFileClip(video_path)
        # 오디오가 있는지 확인
        if video.audio is None:
            print("Error: 이 비디오 파일에는 오디오 트랙이 없습니다!")
            return None
            
        video.audio.write_audiofile(output_path, codec='pcm_s16le') # wav 포맷
        video.close()
        return output_path
        
    except Exception as e:
        print(f"Conversion Failed: {e}")
        return None