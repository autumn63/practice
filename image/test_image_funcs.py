import os
import tempfile
import unittest
from PIL import Image

from src.blur import blur
from src.flip_horizontal import flip_horizontal
from src.flip_vertical import flip_vertical
from src.load_image import load_image
from src.save_image import save_image
from src.crop import crop_image  



#   Load / Save Image Tests

class TestLoadSaveImage(unittest.TestCase):
    def test_save_and_load_roundtrip(self):
        """save_image 후 load_image 했을 때 크기/모드가 유지되는지 확인기 위해"""
        img = Image.new("RGB", (30, 40), (10, 20, 30))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.png")
            save_image(img, path)
            loaded = load_image(path)

        self.assertEqual(loaded.size, img.size)
        self.assertEqual(loaded.mode, img.mode)

    def test_load_invalid_path_raises(self):
        """존재하지 않는 파일을 load_image에 넣었을 때 예외가 발생해야 함"""
        with self.assertRaises(Exception):
            _ = load_image("path/that/does/not/exist.png")



#     Flip Horizontal Tests
class TestFlipHorizontal(unittest.TestCase):
    def test_flip_horizontal_reverses_left_right(self):
        """flip_horizontal이 좌우를 제대로 뒤집는지 확인하기 위해"""
        img = Image.new("RGB", (2, 1))
        img.putpixel((0, 0), (255, 0, 0))  # left = red
        img.putpixel((1, 0), (0, 0, 255))  # right = blue

        flipped = flip_horizontal(img)

        self.assertEqual(flipped.getpixel((0, 0)), (0, 0, 255))
        self.assertEqual(flipped.getpixel((1, 0)), (255, 0, 0))

    def test_flip_horizontal_not_modify_original(self):
        """flip_horizontal이 원본 이미지를 수정하지 않는지 확인하기 위해"""
        img = Image.new("L", (5, 5), 128)
        original = img.copy()

        _ = flip_horizontal(img)  # 결과는 무시

        self.assertEqual(list(img.getdata()), list(original.getdata()))



# Flip Vertical Tests

class TestFlipVertical(unittest.TestCase):
    def test_flip_vertical_reverses_top_bottom(self):
        """flip_vertical이 상하를 제대로 뒤집는지 확인하기 위해"""
        img = Image.new("RGB", (1, 2))
        img.putpixel((0, 0), (255, 0, 0))  # top red
        img.putpixel((0, 1), (0, 0, 255))  # bottom blue

        flipped = flip_vertical(img)

        self.assertEqual(flipped.getpixel((0, 0)), (0, 0, 255))
        self.assertEqual(flipped.getpixel((0, 1)), (255, 0, 0))


# blur tests
class TestBlur(unittest.TestCase):
    def test_blur_keeps_size(self):
        """blur 수행 후에도 이미지 크기는 유지되어야 함"""
        img = Image.new("RGB", (50, 50), (100, 100, 100))
        blurred = blur(img, ksize=3)   
        self.assertEqual(blurred.size, img.size)

    def test_blur_changes_boundary_pixels(self):
        """
        blur 적용 시 경계값의 Gray 변화가 발생해야 한다.
        (흑백 0/255 경계에서 중간 값이 생기는지 확인)
        """
        img = Image.new("L", (10, 10), 0)
        for x in range(5, 10):
            for y in range(10):
                img.putpixel((x, y), 255)

        blurred = blur(img, ksize=5)
        v = blurred.getpixel((5, 5))
        self.assertTrue(0 < v < 255, f"expected gray-ish value, got {v}")


#crop tests
class TestCrop(unittest.TestCase):
    def setUp(self):
        self.img = Image.new("RGB", (100, 80), (255, 0, 0))

    def test_crop_basic_region(self):
        """기본적인 사각형 영역 crop이 원하는 크기로 잘리는지"""
        cropped = crop_image(self.img, 10, 20, 60, 50)
        self.assertEqual(cropped.size, (50, 30))

    def test_crop_out_of_range_raises(self):
        """이미지 범위를 벗어난 crop 요청 시 예외 발생 여부"""
        with self.assertRaises(Exception):
            _ = crop_image(self.img, -10, -10, 200, 200)


if __name__ == "__main__":
    unittest.main(verbosity=2)
