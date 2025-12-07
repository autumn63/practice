#cd data
#python face_detection.py

# pip install mediapipe opencv-python 필요.
import cv2
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

#---- 1. 동영상 파일 열기 -----
cap = cv2.VideoCapture(
    "video/data_video/final_output.mp4"
)

success, frame = cap.read()
if not success:
    print("영상 파일을 찾지 못했습니다.")
    exit()

h,w,c, = frame.shape
fps = cap.get(cv2.CAP_PROP_FPS)

print("원본 해상도:", w, "x", h)
print("원본 FPS:", fps)

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  #처음 프레임으로 되돌리기

#---- 2.fourcc 인코딩 방식 설정. MP4V로 설정.----

fourcc = cv2.VideoWriter_fourcc(*'mp4v')

#동영상 파일 생성. 폴더 위치 설정.
out = cv2.VideoWriter("/Users/heoyeonbeom/Documents/GitHub/practice/video/data_video/output_blur.mp4",
    fourcc,
    cap.get(cv2.CAP_PROP_FPS), #원본 FPS 유지
    (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), #원본 해상도 유지
     int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))) #원본 해상도 유지
)

#---- 3. Mediapipe 설정. -----

with mp_face_detection.FaceDetection(
    model_selection=1,  # 0: 가까운 거리(2m 이내) , 1: 먼 거리(2m 이상)
    min_detection_confidence=0.1  # 최소 얼굴 감지 신뢰도. 낮을수록 애매한 것도 잘 잡음.
) as face_detection:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("영상을 찾지 못했습니다.")
            continue  

        # ---- 4. 전처리 -----

        alpha = 1.00  # 대비 조절 (1.0-3.0)
        beta = 0     # 밝기 조절 (0-100)
        image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)  # 영상의 대비와 밝기 조절

        h, w, _ = image.shape #원본 프레임 사이즈

        # --- Mediapipe용 RGB (원본 image는 그대로 BGR 유지) ---
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = face_detection.process(image_rgb)
        image_rgb.flags.writeable = True

        # ---- 5. 얼굴 감지 -----
        detections = []  # 감지된 얼굴 박스 좌표를 저장할 리스트
        if results.detections:
            for det in results.detections:  # det: 얼굴 감지 객체
                bbox = det.location_data.relative_bounding_box  # 얼굴 경계 상자 정보

                # xmin, ymin: 경계 상자의 왼쪽 위 모서리 좌표 (비율)
                # width, height: 경계 상자의 너비와 높이 (비율)
                # w: 영상 가로 길이(픽셀수)
                # h: 영상 세로 길이(픽셀수)

                x1 = int(bbox.xmin * w)                  # 왼쪽 위 X좌표
                y1 = int(bbox.ymin * h)                  # 왼쪽 위 Y좌표
                x2 = int((bbox.xmin + bbox.width) * w)   # 오른쪽 아래 X좌표
                y2 = int((bbox.ymin + bbox.height) * h)  # 오른쪽 아래 Y좌표

                # 박스 좌표
                #x1,y1-----------x2,y1
                #  |                 |
                #  |                 |
                #  |      얼굴       |
                #  |                 |
                #x1,y2-----------x2,y2
                

                pad = 0.15  # 15% 박스 크기 확장.
                dx = int((x2 - x1) * pad)  # 박스 가로길이 기준 확장량
                dy = int((y2 - y1) * pad)  # 박스 세로길이 기준 확장량

                x1 -= dx  # 왼쪽으로 15% 확장
                y1 -= dy  # 위로 15% 확장
                x2 += dx  # 오른쪽으로 15% 확장
                y2 += dy  # 아래로 15% 확장

                # 박스가 영상 크기를 벗어나지 않도록 조정. (클램프)
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)

                detections.append((x1, y1, x2, y2))  # 감지된 얼굴 박스 좌표 추가

        # ---- 6. 블러 처리 ----
        for (x1, y1, x2, y2) in detections:
            face_region = image[y1:y2, x1:x2]

            if face_region.size == 0:
                continue

            # 얼굴 부분만 가우시안 블러
            blurred = cv2.GaussianBlur(face_region, (71, 71), 35)
            image[y1:y2, x1:x2] = blurred

        # 얼굴 감지 박스를 보고 싶으면 아래 주석 해제
        # if results.detections:
        #     for detection in results.detections:
        #         mp_drawing.draw_detection(image, detection)

        #---- 7. 화면에는 축소분만 표시 ----
        display_w = 1280
        scale = display_w / w
        display_h = int(h * scale)
        image = cv2.resize(image, (display_w, display_h))

        cv2.imshow("Face Blur", image)

        #---- 저장 ----
        out.write(image)

        if cv2.waitKey(5) & 0xFF == 27:  # esc 키를 누르면 종료.
            break

cap.release()
out.release()
cv2.destroyAllWindows()
