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
    w, h = img.size

    center_x = w // 2
    center_y = h // 2

    half = size // 2
    left   = center_x - half
    upper  = center_y - half
    right  = center_x + half
    lower  = center_y + half

    return img.crop((left, upper, right, lower))

