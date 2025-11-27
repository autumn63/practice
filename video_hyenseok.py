"""
video_preprocess.py

영상 전처리 기본 함수 모음

필요 패키지 설치:
    pip install opencv-python

이 파일에서 제공하는 기능:
    1) 해상도 변경         : resize_video
    2) FPS 변경           : change_fps
    3) 특정 구간 자르기    : trim_video
    4) 일정 시간 단위 분할 : split_video
    5) 프레임 이미지 추출  : extract_frames
    6) 흑백(그레이스케일) : to_gray_video
    7) 밝기 조절          : adjust_brightness_video
    8) 좌우 반전          : horizontal_flip_video
    9) 블러(흐리게)        : blur_video
"""

import cv2  
import os    
import math  


# --------------------------------------------------
# 내부 함수
# --------------------------------------------------
def _open_video(path):

    """
    영상 파일을 열어 cv2.VideoCapture 객체를 리턴하는 함수.
    - 파일 존재 여부와 열기 여부를 한 번에 체크한다.
    """
    
    if not os.path.exists(path):
        # 파일이 없으면 에러 발생
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")

    cap = cv2.VideoCapture(path)  # 영상 읽기 객체 생성
    if not cap.isOpened():
        # 파일은 있는데 OpenCV가 못 여는 경우 에러발생
        raise IOError(f"영상 파일을 열 수 없습니다: {path}")
    return cap


def _make_writer(path, fps, width, height, is_color=True, codec="mp4v"):

    """
    출력 영상을 저장할 VideoWriter 객체를 만들어주는 함수.
    - path   : 저장할 파일 경로 (예: 'output.mp4')
    - fps    : 초당 프레임 수
    - width  : 영상 가로 길이
    - height : 영상 세로 길이
    - is_color : 컬러 영상 여부 (흑백이면 False)
    - codec : 비디오 코덱 (기본 'mp4v' → .mp4에 잘 맞음)
    """

    fourcc = cv2.VideoWriter_fourcc(*codec)  # 코덱 설정
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height),
                             isColor=is_color)
    if not writer.isOpened():
        raise IOError(f"출력 영상을 만들 수 없습니다: {path}")
    return writer


# --------------------------------------------------
# 1. 해상도 / FPS 관련 함수
# --------------------------------------------------
def resize_video(input_video, output_video, width=640, height=360):

    """
    영상 해상도를 (width, height)로 바꿔서 새 파일로 저장하는 함수.

    예:
        resize_video("input.mp4", "out_resize.mp4", 640, 360)
    """

    cap = _open_video(input_video)          # 원본 영상 열기
    fps = cap.get(cv2.CAP_PROP_FPS)         # 원본 FPS 가져오기
    if fps == 0:
        # FPS 정보를 못 읽을 수 있음. 기본값 30으로 설정
        fps = 30

    # 출력 영상 저장용 객체 생성
    writer = _make_writer(output_video, fps, width, height, is_color=True)

    while True:
        ret, frame = cap.read()  # 한 프레임씩 읽기
        if not ret:
            # 더 이상 읽을 프레임이 없으면 반복 종료
            break

        # 프레임 크기를 (width, height)로 변경
        resized = cv2.resize(frame, (width, height))
        writer.write(resized)  # 출력 영상에 프레임 추가

    cap.release()    # 입력 영상 리소스 해제
    writer.release() # 출력 영상 저장 마무리

    print(f"[resize_video] 저장 완료: {output_video}")


def change_fps(input_video, output_video, target_fps=15):

    """
    영상의 FPS를 바꾸는 함수 (간단한 샘플링 방식).

    동작 방식:
        - 원본 fps를 orig_fps라고 할 때,
          frame_interval = orig_fps / target_fps 를 계산한다.
        - 그 후, frame_idx % frame_interval == 0 인 프레임만 골라서 쓴다.
        - 즉, 일정 간격으로 프레임을 건너뛰면서 저장하는 방식.

    주의:
        - target_fps < orig_fps : 일부 프레임만 사용 → 영상이 '빠르게' 줄어듦
        - target_fps > orig_fps : 같은 프레임이 여러 번 복제될 수 있음
    """
    
    cap = _open_video(input_video)
    orig_fps = cap.get(cv2.CAP_PROP_FPS)  # 원래 FPS
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if orig_fps == 0:
        raise ValueError("원본 FPS를 읽을 수 없습니다.")

    # 예: orig_fps=30, target_fps=15 → frame_interval=2
    frame_interval = max(int(round(orig_fps / target_fps)), 1)

    # 새로운 FPS로 출력 영상 저장
    writer = _make_writer(output_video, target_fps, width, height,
                          is_color=True)

    frame_idx = 0  # 읽은 프레임 번호 (0부터 시작)
    kept = 0       # 실제로 저장한 프레임 개수

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # frame_interval 간격으로 프레임 선택
        if frame_idx % frame_interval == 0:
            writer.write(frame)
            kept += 1

        frame_idx += 1

    cap.release()
    writer.release()
    print(f"[change_fps] 저장 완료: {output_video} (저장된 프레임 수: {kept})")


# --------------------------------------------------
# 2. 구간 자르기 / 분할
# --------------------------------------------------
def trim_video(input_video, output_video, start_sec=0, end_sec=None):

    """
    영상에서 start_sec ~ end_sec 구간만 잘라서 새로운 영상으로 저장하는 함수.

    매개변수:
        input_video : 원본 영상 파일 경로
        output_video: 잘라낸 구간을 저장할 파일 경로
        start_sec   : 시작 시간(초)
        end_sec     : 끝 시간(초). None이면 끝까지

    예:
        trim_video("input.mp4", "trim_3_8.mp4", start_sec=3, end_sec=8)
    """

    cap = _open_video(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # end_sec가 주어지지 않으면 전체 길이를 사용
    if end_sec is None:
        end_sec = total_frames / fps

    # 초 → 프레임 번호로 변환
    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)

    # 범위 보정 (0 ~ total_frames 사이로 제한)
    start_frame = max(0, start_frame)
    end_frame = min(total_frames, end_frame)

    if start_frame >= end_frame:
        raise ValueError("start_sec과 end_sec 설정이 잘못되었습니다.")

    # 시작 프레임 위치로 이동
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # 잘라낸 구간을 저장할 VideoWriter
    writer = _make_writer(output_video, fps, width, height, is_color=True)

    # 필요한 프레임 수 만큼 반복
    for _ in range(end_frame - start_frame):
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)

    cap.release()
    writer.release()
    print(f"[trim_video] {start_sec:.2f}s ~ {end_sec:.2f}s 구간 저장 완료: {output_video}")


def split_video(input_video, output_dir, chunk_sec=5):
    """
    영상을 chunk_sec 초 단위로 잘라서 여러 개의 파일로 나누는 함수.

    매개변수:
        input_video : 원본 영상
        output_dir  : 잘라낸 여러 영상을 저장할 폴더 이름
        chunk_sec   : 한 조각의 길이(초)

    예:
        30초 영상에서 chunk_sec=5이면
        → chunk_001.mp4 ~ chunk_006.mp4 총 6개 파일이 생성된다.
    """
    # 출력 폴더가 없으면 새로 만든다.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    cap = _open_video(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 한 조각당 들어갈 프레임 수
    frames_per_chunk = int(chunk_sec * fps)
    if frames_per_chunk <= 0:
        raise ValueError("chunk_sec가 너무 작습니다.")

    # 총 몇 개 파일로 나뉘는지 계산 (올림 사용)
    num_chunks = math.ceil(total_frames / frames_per_chunk)
    print(f"[split_video] 총 {num_chunks}개로 분할 예정")

    chunk_idx = 0  # 현재 몇 번째 조각인지
    writer = None  # 현재 조각에 쓰는 VideoWriter
    frame_idx = 0  # 전체 영상 기준 프레임 번호

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 새로운 조각 시작 시점이면 새 파일 생성
        if frame_idx % frames_per_chunk == 0:
            # 이전 writer가 있으면 닫기
            if writer is not None:
                writer.release()

            chunk_idx += 1
            out_path = os.path.join(output_dir,
                                    f"chunk_{chunk_idx:03d}.mp4")
            writer = _make_writer(out_path, fps, width, height,
                                  is_color=True)

        # 현재 조각에 프레임 추가
        writer.write(frame)
        frame_idx += 1

    # 마지막 writer 정리
    if writer is not None:
        writer.release()
    cap.release()
    print(f"[split_video] 분할 완료. 출력 폴더: {output_dir}")


# --------------------------------------------------
# 3. 프레임 이미지 추출
# --------------------------------------------------
def extract_frames(input_video, output_dir, every_n_frames=10, prefix="frame"):

    """
    영상에서 every_n_frames마다 한 장씩 이미지 파일로 저장하는 함수.

    매개변수:
        input_video    : 원본 영상 파일
        output_dir     : 이미지를 저장할 폴더
        every_n_frames : 몇 프레임마다 한 장을 저장할지 (간격)
        prefix         : 파일 이름 앞부분에 붙일 문자열

    예:
        every_n_frames=10이면
        0, 10, 20, 30, ... 번째 프레임을 이미지로 저장한다.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    cap = _open_video(input_video)

    frame_idx = 0  # 읽은 프레임 번호
    saved = 0      # 실제 저장한 이미지 개수

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % every_n_frames == 0:
            filename = f"{prefix}_{frame_idx:06d}.jpg"
            out_path = os.path.join(output_dir, filename)
            cv2.imwrite(out_path, frame)  # 이미지 파일로 저장
            saved += 1

        frame_idx += 1

    cap.release()
    print(f"[extract_frames] 총 {saved}장 저장 완료. 폴더: {output_dir}")


# --------------------------------------------------
# 4. 색/밝기/흑백/블러 관련
# --------------------------------------------------
def to_gray_video(input_video, output_video):

    """
    컬러 영상을 흑백(그레이스케일, 회색음영) 영상으로 변환해서 저장하는 함수.

    예:
        to_gray_video("input.mp4", "out_gray.mp4")
    """
    
    cap = _open_video(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 흑백 영상이므로 is_color=False
    writer = _make_writer(output_video, fps, width, height,
                          is_color=False)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 컬러(BGR) → 흑백 변환
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        writer.write(gray)

    cap.release()
    writer.release()
    print(f"[to_gray_video] 저장 완료: {output_video}")


def adjust_brightness_video(input_video, output_video, factor=1.2):

    """
    영상의 밝기를 조절하는 함수.

    매개변수:
        factor > 1.0 : 더 밝게
        factor < 1.0 : 더 어둡게

    내부적으로는 cv2.convertScaleAbs를 사용해서
    각 픽셀 값에 factor를 곱해준다.
    """

    cap = _open_video(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = _make_writer(output_video, fps, width, height,
                          is_color=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # alpha=factor, beta=0 → 밝기만 조절 (대비/오프셋 X)
        out = cv2.convertScaleAbs(frame, alpha=factor, beta=0)
        writer.write(out)

    cap.release()
    writer.release()
    print(f"[adjust_brightness_video] factor={factor}, 저장 완료: {output_video}")


# --------------------------------------------------
# 5. 좌우 반전 / 블러
# --------------------------------------------------
def horizontal_flip_video(input_video, output_video):

    """
    영상을 좌우로 뒤집어서(거울처럼) 저장하는 함수.

    예:
        horizontal_flip_video("input.mp4", "out_flip.mp4")
    """

    cap = _open_video(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = _make_writer(output_video, fps, width, height,
                          is_color=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # flipCode=1 → 좌우 반전
        flipped = cv2.flip(frame, 1)
        writer.write(flipped)

    cap.release()
    writer.release()
    print(f"[horizontal_flip_video] 저장 완료: {output_video}")


def blur_video(input_video, output_video, ksize=5):

    """
    영상에 가우시안 블러를 적용해서 전체적으로 흐릿하게 만드는 함수.
    (노이즈 줄이기, 자연스럽게 보이게 하기 등에 사용 가능)

    매개변수:
        ksize : 블러 커널 크기 (반드시 홀수, 예: 3, 5, 7)

    예:
        blur_video("input.mp4", "out_blur.mp4", ksize=5)
    """

    if ksize % 2 == 0:
        # 가우시안 블러는 홀수 크기만 사용 가능
        raise ValueError("ksize는 홀수여야 합니다. 예: 3, 5, 7")

    cap = _open_video(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = _make_writer(output_video, fps, width, height,
                          is_color=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # (ksize, ksize) 크기의 가우시안 커널로 블러 적용
        blurred = cv2.GaussianBlur(frame, (ksize, ksize), 0)
        writer.write(blurred)

    cap.release()
    writer.release()
    print(f"[blur_video] ksize={ksize}, 저장 완료: {output_video}")


# --------------------------------------------------
# 이 파일을 직접 실행했을 때만 동작하는 테스트용 코드
# (다른 파일에서 import해서 쓸 때는 실행되지 않는다.)
# --------------------------------------------------
if __name__ == "__main__":
    # 테스트용으로 사용할 입력 파일 이름
    src = "input.mp4"

    # 필요할 때 아래 주석을 풀고 하나씩 테스트하면 된다.
    # 실제 과제 코드에서는 이 부분은 안 써도 됨.

    # 1. 해상도 변경
    # resize_video(src, "out_resize.mp4", width=640, height=360)

    # 2. FPS 변경
    # change_fps(src, "out_fps15.mp4", target_fps=15)

    # 3. 3초~8초만 잘라내기
    # trim_video(src, "out_trim_3_8.mp4", start_sec=3, end_sec=8)

    # 4. 5초 단위로 나누기
    # split_video(src, "splits", chunk_sec=5)

    # 5. 10프레임마다 한 장씩 이미지로 저장
    # extract_frames(src, "frames", every_n_frames=10)

    # 6. 흑백 영상 만들기
    # to_gray_video(src, "out_gray.mp4")

    # 7. 밝기 조절 (조금 더 밝게)
    # adjust_brightness_video(src, "out_bright.mp4", factor=1.3)

    # 8. 좌우 반전
    # horizontal_flip_video(src, "out_flip.mp4")

    # 9. 블러 적용
    # blur_video(src, "out_blur.mp4", ksize=5)

    print("이 파일은 전처리 함수 모음입니다. 다른 코드에서 import 해서 사용하세요.")
