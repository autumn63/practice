from PIL import Image


def flip_horizontal(img):
    return img.transpose(Image.FLIP_LEFT_RIGHT) #
