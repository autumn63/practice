# python -m pip install opencv-python 필요.

import cv2
#cv2.VideoCapture(index) index = 0 노트북 내장카메라  index = 1~n 외장카메라
capture = cv2.VideoCapture(0) #카메라 열기

#capture.set(propid, value) propid: 설정할 속성, value: 설정할 값
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640) #가로 크기 설정
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) #세로 크기 설정

if not capture.isOpened():
    print("Camera open failed!") # 열리지 않았을때 출력
    exit()

capture.release() #카메라 닫기 메모리 해제
cv2.destroyAllWindows() #모든 윈도우 창 제거.  cv2.destroyWindow(winname) #특정 창 제거.