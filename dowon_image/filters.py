def blur(img, ksize=3):
    """
    이미지를 블러(흐리게) 처리하는 함수.
    TODO: 복잡한 커널 설계 대신, Pillow에서 제공하는
          ImageFilter.GaussianBlur(radius)를 그대로 사용하는 방향으로 구현 예정.
          radius 값으로 흐림 정도만 조절하고, 커널 크기를 직접 다루지는 않음.
    """
    pass

def sharpen(img):
    """
    이미지를 선명하게(샤프닝) 만드는 함수.
    TODO: 샤프닝 커널을 직접 설계하지 않고,
          Pillow의 ImageFilter.SHARPEN 필터를 사용하는 단순한 방식으로 구현 예정.
    """
    pass

def edge_detect(img, method="sobel"):
    """
    이미지의 윤곽(엣지)을 강조하는 함수.
    TODO: Sobel, Canny 같은 복잡한 알고리즘은 구현하지 않고,
          Pillow의 ImageFilter.FIND_EDGES나 EDGE_ENHANCE 계열 필터를
          활용하는 수준에서만 사용 예정.
    """
    pass

def adjust_brightness_contrast(img, brightness=0.0, contrast=0.0):
    """
    이미지의 밝기와 대비를 조절하는 함수.
    TODO: 픽셀 값에 선형 변환을 적용해 밝기/대비를 조절하는 로직 구현 예정.
          (예: output = img * (1 + contrast) + brightness 형태로 처리)
    """
    pass

