from PIL import Image

def to_gray(img):
    gray = img.convert("L")
    return gray

def resize(img, width, height, keep_ratio=False):
    if keep_ratio:
      
      original_w, original_h = img.size
      ratio = min(width / original_w, height / original_h)
      
      new_w = int(original_w * ratio)
      new_h = int(original_h * ratio)
        
      return img.resize((new_w, new_h), Image.LANCZOS)
      
      
    return img.resize((width, height), Image.LANCZOS)
    
def flip_horizontal(img):
    return img.transpose(Image.FLIP_LEFT_RIGHT) #


def flip_vertical(img):
    return img.transpose(Image.FLIP_TOP_BOTTOM)


def rotate(img, angle, expand=True):
    return img.rotate(angle, expand=expand)

def center_crop(img, size):
    """
    이미지를 중심 기준으로 정사각형(size x size) 크기로 잘라내는 함수.
    TODO: 이미지의 중앙 좌표를 계산하여 crop 영역 산출 후
          PIL의 img.crop(box) 사용해 잘라내기 로직 구현 예정.
    """
    pass
