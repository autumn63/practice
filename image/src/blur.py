from PIL import ImageFilter

def blur(img, ksize=3):

    # ksize는 직접 사용안함.. 흐림 정도를 radius로만 반영
    radius = ksize / 2
    return img.filter(ImageFilter.GaussianBlur(radius=radius))
