'''
load_iamge(path)
로컬 경로에서 이미지를 읽어 PIL.Image.Image 객체로 반환
RGB로 통일해서 반환 → 후처리 함수들이 일관성 있게 처리 가능

save_image(path)
Image 객체를 지정한 경로에 저장
확장자 자동감지 (jpg / png 등)
'''

from PIL import Image
import os

def load_image(path):

    if not os.path.exists(path):
        raise FileNotFoundError(f"[load_image] File not found:{path}")
    
    img = Image.open(path).convert("RGB")
    return img

def save_image(img, path):

    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    img.save(path)
    print(f"[save_image] Image saved to: {path}")
