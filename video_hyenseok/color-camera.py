'''
pip install opencv-python numpy matplotlib
'''
#카메라가 이미 다른 앱에서 쓰고 있으면 열리지 않을 수 있다.
#개인 정보 및 보안 → 카메라 → 카메라 접근 허용, "데스크톱 앱이 카메라에 접근하도록 허용"

#터미널 입력 cd video_hyenseok , python opencv-camera.py

# 카메라로 영상 읽고, 원본 영상, 흑백 영상 둘다 출력. 원본 영상만 저장.
import cv2
import numpy as np
import matplotlib.pyplot as plt

'''
capture.set(propid, value) propid: 설정할 속성, value: 설정할 값
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640) #가로 크기 설정
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) #세로 크기 설정
'''

capture = cv2.VideoCapture(0) #카메라 열기
#cv2.VideoCapture(index) index = 0 노트북 내장카메라  index = 1~n 외장카메라

if not capture.isOpened():
    print("Camera open failed!") # 열리지 않았을때 출력
    exit()

ret, frame = capture.read()
if not ret:
    print("Frame read failed!")
    capture.release()
    exit()

    
h, w = frame.shape[:2]
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('color.avi', fourcc, 20.0, (w, h))  # 프레임 크기 자동 맞추기

#노란색 HSV 값
YELLOW_H = 29
YELLOW_S = 255
YELLOW_V = 255

while True:
    ret, frame = capture.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 검정색 범위 (조도에 따라 V 값은 조절 가능)
    # H(색상)이 0~180 사이,
    # S(채도)가 0~255 사이,
    # V(밝기)가 0~60 사이인 어두운 부분을 "검정"으로 분류
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 60])

    #mask 조건에 맞는 색 --> 하얀색, 맞지 않는 색 --> 검은색
    mask = cv2.inRange(hsv, lower_black, upper_black)

    # 검정색 영역을 노란색으로 변경
    colored_hsv = hsv.copy()
    colored_hsv[..., 0] = np.where(mask > 0, YELLOW_H, colored_hsv[..., 0])  # H 변경
    colored_hsv[..., 1] = np.where(mask > 0, YELLOW_S, colored_hsv[..., 1])  # S 변경
    colored_hsv[..., 2] = np.where(mask > 0, YELLOW_V, colored_hsv[..., 2])  # V 변경

    # BGR 색상 공간으로 변환
    colored = cv2.cvtColor(colored_hsv, cv2.COLOR_HSV2BGR)
    
    '''
    #res 조건에 맞는색--> 원래색 유지, 맞지 않는 색 --> 검은색
    res = cv2.bitwise_and(frame, frame, mask=mask)
    '''

    out.write(colored) #영상 저장
    cv2.imshow("frame", frame) #원래 영상 출력
    # cv2.imshow("mask", mask)
    cv2.imshow("res", colored) #검정색 부분 노란색으로 바뀐 영상 출력

    if cv2.waitKey(1) == 27: #esc키 누르면 종료
        break


capture.release() #카메라 닫기 메모리 해제
out.release() #영상 저장 닫기 메모리 해제
cv2.destroyAllWindows() #모든 윈도우 창 제거.  cv2.destroyWindow(winname) #특정 창 제거.