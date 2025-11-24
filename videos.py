"""
videos.py

필요 패키지:
    pip install opencv-python
"""

import cv2
from pathlib import Path


# -----------------------------
# 1. 공용 유틸 함수들
# -----------------------------

def open_video_source(video_source):
    """
    영상 입력 소스를 열어서 VideoCapture 객체를 반환하는 함수.

    Parameters
    ----------
    video_source : int or str
        - 0, 1, 2 ... : 웹캠 번호 (보통 0이 기본 웹캠)
        - "cctv.mp4" 같은 문자열 : 영상 파일 경로
        - "rtsp://..." : CCTV RTSP 주소

    Returns
    -------
    video_capture : cv2.VideoCapture
        영상 프레임을 순서대로 읽을 수 있는 OpenCV 객체.
    """
    video_capture = cv2.VideoCapture(video_source)

    if not video_capture.isOpened():
        raise RuntimeError(f"영상 소스를 열 수 없습니다: {video_source}")

    return video_capture


def create_video_writer(output_video_path, frames_per_second, frame_width, frame_height):
    """
    새로운 영상 파일을 저장하기 위한 VideoWriter 객체를 만들어주는 함수.

    Parameters
    ----------
    output_video_path : str or Path
        저장할 출력 영상 파일 경로. (예: 'motion_only.mp4')
    frames_per_second : float
        출력 영상의 FPS 값 (보통 입력 영상의 FPS를 그대로 사용)
    frame_width : int
        영상 가로 길이 (픽셀 수)
    frame_height : int
        영상 세로 길이 (픽셀 수)

    Returns
    -------
    video_writer : cv2.VideoWriter
        frame 단위로 write()를 호출해서 영상으로 저장할 수 있는 객체.
    """
    output_video_path = Path(output_video_path)
    output_video_path.parent.mkdir(parents=True, exist_ok=True)

    # fourcc 는 "어떤 코덱으로 인코딩할지"를 뜻함.
    # "mp4v" 는 mp4 파일을 만들 때 많이 사용하는 기본 코덱 중 하나.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    video_writer = cv2.VideoWriter(
        str(output_video_path),
        fourcc,
        frames_per_second,
        (frame_width, frame_height)
    )

    if not video_writer.isOpened():
        raise RuntimeError(f"영상 파일을 만들 수 없습니다: {output_video_path}")

    return video_writer


# -----------------------------
# 2. 사람(보행자) 검출 설정
# -----------------------------

# HOG + SVM 기반 사람 검출기 (OpenCV에서 기본 제공하는 보행자 detector)
people_detector = cv2.HOGDescriptor()
people_detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


def detect_people_in_frame(color_frame):
    """
    한 프레임(컬러 영상)에서 사람을 찾아내는 함수.

    Parameters
    ----------
    color_frame : numpy.ndarray
        BGR 색상 채널 순서의 영상 프레임 (cv2.VideoCapture로 읽어온 그대로)

    Returns
    -------
    person_boxes : list of tuple
        (x, y, w, h) 형태의 bounding box 리스트.
        x, y 는 좌상단 좌표, w, h 는 너비와 높이.
    person_count : int
        검출된 사람 수.
    """
    # HOG detector는 컬러 그대로 넣어도 동작하지만
    # 보통 크기를 조금 줄여서 속도를 올리기도 한다.
    # 여기서는 이해하기 쉽게 그대로 사용.
    found_rectangles, found_weights = people_detector.detectMultiScale(
        color_frame,
        winStride=(8, 8),   # 슬라이딩 윈도우 이동 간격
        padding=(8, 8),     # 주변 패딩
        scale=1.05          # 이미지 피라미드 스케일
    )

    person_boxes = []
    for (x, y, w, h) in found_rectangles:
        person_boxes.append((x, y, w, h))

    person_count = len(person_boxes)
    return person_boxes, person_count


# -----------------------------
# 3. CCTV 영상에서 사람 유무 + 움직임 구간 추출
# -----------------------------

def extract_motion_video_from_cctv(
    input_video_source,
    output_motion_video_path,
    motion_pixel_ratio_threshold=0.02,
    frame_difference_threshold=25,
    tail_frames_after_motion=15
):
    """
    CCTV 영상에서 "사람이 있는" 프레임만 골라서
    하나의 새로운 영상으로 저장하는 함수.

    기본 아이디어:
    - 연속된 두 프레임의 차이를 보고 "얼마나 많이 변했는지" 계산
    - 프레임의 일정 비율 이상 픽셀이 변하면 → 움직임이 있다고 판단
    - 움직임이 감지된 프레임만 새 VideoWriter로 write()

    Parameters
    ----------
    input_video_source : int or str
        영상 입력 소스 (웹캠 번호, 파일 경로, RTSP 주소 등)
    output_motion_video_path : str or Path
        움직임이 있는 부분만 모아서 저장할 출력 mp4 경로
    motion_pixel_ratio_threshold : float
        0 ~ 1 사이 값. 전체 픽셀 중 이 비율 이상이 변하면
        "움직임 있음"이라고 판단.
        예: 0.02 = 전체 픽셀의 2% 이상이 변한 경우.
    frame_difference_threshold : int
        0 ~ 255 사이 값. 프레임 간 차이에서 이 값 이상일 때
        "변한 픽셀"로 간주.
    tail_frames_after_motion : int
        움직임이 멈춘 뒤에도 몇 프레임 더 이어서 저장할지.
        너무 딱 끊어지지 않게 약간의 여유를 줌.

    Returns
    -------
    motion_segments : list of tuple
        (start_frame_index, end_frame_index) 형태의 리스트.
        움직임이 있었던 구간들의 프레임 인덱스 범위.
    """
    # 1) 입력 영상 열기
    video_capture = open_video_source(input_video_source)

    frames_per_second = video_capture.get(cv2.CAP_PROP_FPS)
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if frames_per_second == 0:
        video_capture.release()
        raise ValueError("입력 영상의 FPS 정보를 읽을 수 없습니다.")

    # 2) 출력용 VideoWriter 생성
    motion_video_writer = create_video_writer(
        output_motion_video_path,
        frames_per_second,
        frame_width,
        frame_height
    )

    # 3) 첫 프레임을 읽어서 "이전 프레임" 기준으로 사용
    has_frame, previous_color_frame = video_capture.read()
    if not has_frame:
        video_capture.release()
        motion_video_writer.release()
        raise ValueError("입력 영상에서 첫 프레임을 읽을 수 없습니다.")

    # 프레임을 흑백으로 바꾸고, 가우시안 블러를 걸어서 노이즈 감소
    previous_gray_frame = cv2.cvtColor(previous_color_frame, cv2.COLOR_BGR2GRAY)
    previous_gray_frame = cv2.GaussianBlur(previous_gray_frame, (5, 5), 0)

    total_pixels = previous_gray_frame.shape[0] * previous_gray_frame.shape[1]

    # 움직임 구간 정보를 저장하기 위한 변수들
    is_currently_in_motion = False          # 지금이 "움직임 구간" 안인지 여부
    remaining_tail_frames = 0               # 움직임 종료 후 tail로 남은 프레임 수
    motion_segments = []                    # (start_frame, end_frame) 리스트
    current_segment_start_index = None      # 현재 구간의 시작 프레임 인덱스

    current_frame_index = 1  # 위에서 이미 첫 프레임을 읽었으므로 1부터 시작

    # 4) 나머지 프레임들 반복 처리
    while True:
        has_frame, current_color_frame = video_capture.read()
        if not has_frame:
            break  # 더 이상 읽을 프레임이 없으면 반복 종료

        # 현재 프레임도 흑백 + 블러 처리
        current_gray_frame = cv2.cvtColor(current_color_frame, cv2.COLOR_BGR2GRAY)
        current_gray_frame = cv2.GaussianBlur(current_gray_frame, (5, 5), 0)

        # 이전 프레임과 현재 프레임의 차이 계산
        frame_difference = cv2.absdiff(previous_gray_frame, current_gray_frame)

        # frame_difference_threshold 보다 큰 픽셀만 남기기 (이진 이미지)
        # → 차이가 작으면 0, 크면 255 로 처리됨
        _, difference_binary = cv2.threshold(
            frame_difference,
            frame_difference_threshold,
            255,
            cv2.THRESH_BINARY
        )

        # 변한 픽셀 개수 세기
        changed_pixel_count = cv2.countNonZero(difference_binary)

        # 전체 픽셀 대비 변한 픽셀 비율 (0 ~ 1)
        changed_pixel_ratio = changed_pixel_count / float(total_pixels)

        # 움직임 여부 판단
        if changed_pixel_ratio > motion_pixel_ratio_threshold:
            # 이번 프레임에서는 움직임이 있다고 판단

            if not is_currently_in_motion:
                # 이전에는 움직임이 없었는데, 지금 처음 발생한 경우 → 새 구간 시작
                is_currently_in_motion = True
                current_segment_start_index = current_frame_index

            # tail 프레임 카운트 리셋 (나중에 서서히 줄여나감)
            remaining_tail_frames = tail_frames_after_motion

            # 움직임이 있는 프레임은 무조건 저장
            motion_video_writer.write(current_color_frame)

        else:
            # 이번 프레임에서는 변화가 크지 않음 (움직임이 약함 / 없음)
            if is_currently_in_motion:
                if remaining_tail_frames > 0:
                    # tail 프레임 구간: 아직은 구간 안에 포함
                    motion_video_writer.write(current_color_frame)
                    remaining_tail_frames -= 1
                else:
                    # tail도 끝났으므로 구간 종료
                    is_currently_in_motion = False
                    if current_segment_start_index is not None:
                        motion_segments.append(
                            (current_segment_start_index, current_frame_index)
                        )
                    current_segment_start_index = None
            # is_currently_in_motion == False 인 경우는
            # 아무 구간에도 속하지 않으므로 그냥 버린다.

        # 다음 루프에서 사용하기 위해 현재 프레임을 previous로 교체
        previous_gray_frame = current_gray_frame
        current_frame_index += 1

    # 루프가 끝났는데, 여전히 움직임 구간이 열려 있는 경우 마지막 구간 마무리
    if is_currently_in_motion and current_segment_start_index is not None:
        motion_segments.append((current_segment_start_index, current_frame_index - 1))

    # 자원 정리
    video_capture.release()
    motion_video_writer.release()

    return motion_segments


# -----------------------------
# 4. 사람 유무 + 박스 시각화 예시 (실시간 보기용)
# -----------------------------

def show_cctv_with_people_detection(video_source=0):
    """
    CCTV / 웹캠 화면에 사람 검출 결과를 그려서
    실시간으로 보여주는 간단한 데모 함수.

    Parameters
    ----------
    video_source : int or str
        0 → 기본 웹캠
        "cctv.mp4" → 영상 파일
        "rtsp://..." → CCTV 스트림 주소
    """
    video_capture = open_video_source(video_source)

    while True:
        has_frame, color_frame = video_capture.read()
        if not has_frame:
            print("더 이상 읽을 프레임이 없습니다. 종료합니다.")
            break

        # 한 프레임에서 사람 검출
        person_boxes, person_count = detect_people_in_frame(color_frame)

        # 사람 주변에 초록색 박스 그리기
        for (x, y, w, h) in person_boxes:
            cv2.rectangle(color_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 상태 텍스트 만들기
        if person_count > 0:
            status_text = f"Occupied - People: {person_count}"
            status_color = (0, 0, 255)  # 빨간색
        else:
            status_text = "Available - No People"
            status_color = (0, 255, 0)  # 초록색

        # 화면 상단에 상태 박스 + 텍스트 표시
        cv2.rectangle(color_frame, (0, 0), (350, 40), (0, 0, 0), -1)
        cv2.putText(
            color_frame,
            status_text,
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            status_color,
            2,
            cv2.LINE_AA
        )

        cv2.imshow("CCTV People Detection Demo", color_frame)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()


# -----------------------------
# 5. 단독 실행 테스트용
# -----------------------------

if __name__ == "__main__":
    # 1) 실시간 사람 검출 화면 보고 싶을 때 (웹캠 기준)
    # show_cctv_with_people_detection(video_source=0)

    # 2) CCTV 영상에서 움직임 있는 부분만 추출해서 저장하고 싶을 때
    segments = extract_motion_video_from_cctv(
        input_video_source="cctv_sample.mp4",      # 테스트용 CCTV 영상 파일 경로
        output_motion_video_path="motion_only.mp4"
    )
    print("움직임 구간(프레임 단위):", segments)
