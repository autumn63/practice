import os
from moviepy import VideoFileClip

def convert(base_dir, filename):
    # 경로 설정
    input_dir = os.path.join(base_dir, "input")
    output_dir = os.path.join(base_dir, "output")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    video_path = os.path.join(input_dir, filename)
    audio_filename = os.path.splitext(filename)[0] + ".wav"
    output_path = os.path.join(output_dir, audio_filename)

    try:
        # 비디오 로드 및 오디오 추출
        video = VideoFileClip(video_path)
        
        if video.audio is None:
            print("Error: 오디오 트랙이 없는 비디오입니다.")
            return None
            
        # 오디오 저장 (로그 출력 최소화)
        video.audio.write_audiofile(output_path, codec='pcm_s16le', logger=None)
        video.close()
        
        return output_path
        
    except Exception as e:
        print(f"변환 실패: {e}")
        return None