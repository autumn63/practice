import os
from moviepy import VideoFileClip  # 최신 버전 호환성을 위해 수정

def convert(video_name):
    
    try:
        # PATH: 일단 비디오1 변환해보기
        # os.path.join을 사용하여 경로 에러 방지
        video_path = os.path.join("data", "input", video_name)
        
        # 확장자를 안전하게 떼어내고 .mp3 붙이기
        filename_without_ext = os.path.splitext(video_name)[0]
        audio_path = os.path.join("data", "output", filename_without_ext + ".mp3")

        # 비디오 파일 로드
        # with 구문을 써서 메모리 누수방지.
        with VideoFileClip(video_path) as clip:
            # 오디오 추출 및 저장
            clip.audio.write_audiofile(audio_path)

        return audio_path
        
    except Exception as e:
        print("Error during conversion:", e)
        return -1