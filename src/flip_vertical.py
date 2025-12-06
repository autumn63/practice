from PIL import Image

def flip_vertical(img):
    return img.transpose(Image.FLIP_TOP_BOTTOM)
