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

fourcc = cv2.VideoWriter_fourcc(*'XVID') #코덱 설정
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640,480)) #output.avi로 저장됨, 코덱, fps, 프레임크기

if not capture.isOpened():
    print("Camera open failed!") # 열리지 않았을때 출력
    exit()
    
while True:

    ret , frame = capture.read()
    out.write(frame) #영상 저장
    cv2.imshow("frame", frame) #카메라 영상 출력

    if cv2.waitKey(1) == 27: #esc키 누르면 종료
        break

capture.release() #카메라 닫기 메모리 해제
out.release() #영상 저장 닫기 메모리 해제
cv2.destroyAllWindows() #모든 윈도우 창 제거.  cv2.destroyWindow(winname) #특정 창 제거.