from PIL import Image
import os

def save_image(img, path):

    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    img.save(path)
    print(f"[save_image] Image saved to: {path}")