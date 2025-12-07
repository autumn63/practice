from PIL import Image
import os

def load_image(path):

    if not os.path.exists(path):
        raise FileNotFoundError(f"[load_image] File not found:{path}")
    
    img = Image.open(path).convert("RGB")
    return img