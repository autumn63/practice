def blur(img, ksize=3):
    """
    이미지를 블러(흐리게) 처리하는 함수.
    TODO: PIL 또는 OpenCV의 블러/가우시안 블러 기능을 사용해
          커널 크기(ksize)에 따라 서로 다른 정도의 블러를 적용하는 로직 구현 예정.
    """
    pass

def sharpen(img):
    """
    이미지를 선명하게(샤프닝) 만드는 함수.
    TODO: 샤프닝 커널(예: 엣지를 강조하는 컨볼루션 커널)을 적용해
          이미지의 경계 부분을 더 또렷하게 만드는 로직 구현 예정.
    """
    pass

def edge_detect(img, method="sobel"):
    """
    이미지에서 엣지(윤곽선)를 검출하는 함수.
    TODO: 'sobel', 'canny' 등 다양한 엣지 검출 방법을 지원하도록 확장 예정.
          초기에는 가장 구현이 쉬운 방식부터 지원하고, 추후 옵션을 늘릴 계획.
    """
    pass

def adjust_brightness_contrast(img, brightness=0.0, contrast=0.0):
    """
    이미지의 밝기와 대비를 조절하는 함수.
    TODO: 픽셀 값에 선형 변환을 적용해 밝기/대비를 조절하는 로직 구현 예정.
          (예: output = img * (1 + contrast) + brightness 형태로 처리)
    """
    pass

