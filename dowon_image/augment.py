def random_flip(img, p=0.5):
    """
    이미지를 확률적으로 좌우 뒤집기 또는 그대로 두는 데이터 증강 함수.
    TODO: 확률 p에 따라 이미지를 좌우로 뒤집거나(flip_horizontal 재사용 예정)
          원본을 그대로 반환하는 로직 구현 필요.
    """
    pass

def random_rotate(img, max_angle=15):
    """
    -max_angle ~ +max_angle 범위에서 임의의 각도로 이미지를 회전시키는 데이터 증강 함수.
    TODO: random 모듈을 사용해 각도를 샘플링한 뒤,
          basic.py에 정의된 rotate 함수를 재사용하는 방식으로 구현 예정.
    """
    pass

def random_crop(img, crop_size):
    """
    원본 이미지 안에서 임의 위치를 선택해 crop_size x crop_size 크기로 잘라내는 데이터 증강 함수.
    TODO: 이미지의 가로/세로 크기 내에서 랜덤 시작 좌표를 선택하고,
          PIL의 img.crop(box)를 이용해 잘라내는 로직 구현 필요.
    """
    pass

def random_color_jitter(img, brightness=0.2, contrast=0.2, saturation=0.2):
    """
    밝기, 대비, 채도를 랜덤하게 변화시켜 주는 데이터 증강 함수.
    TODO: 각 파라미터 범위 내에서 랜덤 계수를 샘플링한 뒤,
          픽셀 값에 선형 변환을 적용하거나 색 공간 변환을 통해
          밝기/대비/채도 변화 효과를 주는 로직 구현 예정.
    """
    pass
