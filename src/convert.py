import os
from moviepy import VideoFileClip

def convert(video_name):
    try:
        video_path = os.path.join("data", "input", video_name)
        
        filename_without_ext = os.path.splitext(video_name)[0]
        audio_path = os.path.join("data", "output", filename_without_ext + ".wav")

        # 출력 폴더 확인
        os.makedirs(os.path.dirname(audio_path), exist_ok=True)

        # 비디오 로드 및 오디오 추출 
        with VideoFileClip(video_path) as clip:
            clip.audio.write_audiofile(audio_path,codec='pcm_s16le', logger=None)
            
        return audio_path
        
    except Exception:
        return -1