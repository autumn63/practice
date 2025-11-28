from PIL import Image


def to_gray(img):
    gray = img.convert("L")
    return gray

from PIL import Image

def resize(img, width, height, keep_ratio=False):
    if keep_ratio:
        original_w, original_h = img.size
        ratio = min(width / original_w, height / original_h)

        new_w = int(original_w * ratio)
        new_h = int(original_h * ratio)

        return img.resize((new_w, new_h), Image.LANCZOS)

    return img.resize((width, height), Image.LANCZOS)

def normalize(img, method="0_1"):
    """
    이미지 픽셀 값을 정규화하는 함수.
    TODO: 다양한 정규화 방식 지원 예정:
        - '0_1'      : [0,1] 범위로 스케일링
        - '-1_1'     : [-1,1] 범위로 스케일링
        - 'mean_std' : 평균/표준편차 기반 정규화
    """
    pass

def flip_horizontal(img):
    """
    이미지를 좌우로 뒤집는 함수.
    TODO: PIL의 Image.FLIP_LEFT_RIGHT 등을 사용해 실제 뒤집기 로직 구현 예정.
    """
    pass

def flip_vertical(img):
    """
    이미지를 상하로 뒤집는 함수.
    TODO: PIL의 Image.FLIP_TOP_BOTTOM 등을 사용해 뒤집기 로직 구현 필요.
    """
    pass

def rotate(img, angle, expand=True):
    """
    이미지를 지정한 각도(angle)만큼 회전시키는 함수.
    TODO: PIL의 img.rotate(angle, expand=expand) 등을 이용해 회전 기능 구현 예정.
          추후 90/180/270도 고정 회전도 별도 옵션으로 지원 가능.
    """
    pass

def center_crop(img, size):
    """
    이미지를 중심 기준으로 정사각형(size x size) 크기로 잘라내는 함수.
    TODO: 이미지의 중앙 좌표를 계산하여 crop 영역 산출 후
          PIL의 img.crop(box) 사용해 잘라내기 로직 구현 예정.
    """
    pass

