from PIL import Image

def resize(img, width, height, keep_ratio=False):
    if keep_ratio:
      
      original_w, original_h = img.size
      ratio = min(width / original_w, height / original_h)
      
      new_w = int(original_w * ratio)
      new_h = int(original_h * ratio)
        
      return img.resize((new_w, new_h), Image.LANCZOS)
      
      
    return img.resize((width, height), Image.LANCZOS)
    