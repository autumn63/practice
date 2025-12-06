# main.py
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import ImageTk

from src.load_image import load_image
from src.save_image import save_image
from src.blur import blur
from src.flip_horizontal import flip_horizontal
from src.flip_vertical import flip_vertical
from src.crop import crop_image


class ImageToolApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Tool - Blur / Crop / Flip / Undo")

        # 현재 이미지
        self.img = None
        self.tk_img = None
        self.image_id = None

        # Undo 히스토리
        self.history = []

        # 드래그를 위한 코드
        self.start_x = None
        self.start_y = None
        self.cur_rect = None

        # 모드: "blur" 또는 "crop"
        self.mode = "blur"  # 기본은 blur 드래그

        # 상단 버튼 구현..
        top = tk.Frame(self.root)
        top.pack(side=tk.TOP, fill=tk.X)

        tk.Button(top, text="이미지 열기", command=self.open_image).pack(side=tk.LEFT, padx=5, pady=5)
        tk.Button(top, text="이미지 저장", command=self.save_current_image).pack(side=tk.LEFT, padx=5, pady=5)

        # 상단 버튼들 기능구현 코드
        func = tk.Frame(self.root)
        func.pack(side=tk.TOP, fill=tk.X)

        tk.Button(func, text="좌우 반전", command=self.do_flip_h).pack(side=tk.LEFT, padx=5, pady=5)
        tk.Button(func, text="상하 반전", command=self.do_flip_v).pack(side=tk.LEFT, padx=5, pady=5)
        tk.Button(func, text="전체 Blur", command=self.do_blur_all).pack(side=tk.LEFT, padx=5, pady=5)

        # 드래그해서 crop하기 구현
        tk.Button(func, text="드래그 자르기 (Crop 모드)", command=self.enable_drag_crop).pack(side=tk.LEFT, padx=5, pady=5)

        tk.Button(func, text="되돌리기 (Undo)", command=self.undo).pack(side=tk.LEFT, padx=5, pady=5)

        self.mode_label = tk.Label(func, text="현재 모드: Blur (드래그 시 해당 영역만 블러)")
        self.mode_label.pack(side=tk.LEFT, padx=10)

        # 이미지 작업할 캔버스
        self.canvas = tk.Canvas(self.root, bg="gray")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # 마우스 이벤트
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

    # 공통
    def update_canvas(self):
        if self.img is None:
            return

        self.tk_img = ImageTk.PhotoImage(self.img)

        if self.image_id is None:
            self.image_id = self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)
        else:
            self.canvas.itemconfig(self.image_id, image=self.tk_img)

        self.canvas.config(width=self.img.width, height=self.img.height)

    def push_history(self):
        if self.img is not None:
            self.history.append(self.img.copy())

    def undo(self):
        if not self.history:
            messagebox.showinfo("Undo", "되돌릴 작업이 없습니다.")
            return
        self.img = self.history.pop()
        self.update_canvas()

    # 파일 열기, 저장
    def open_image(self):
        path = filedialog.askopenfilename(
            title="이미지 선택",
            filetypes=[
                ("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return

        self.img = load_image(path)
        self.history.clear()
        self.mode = "blur"
        self.mode_label.config(text="현재 모드: Blur (드래그 시 해당 영역만 블러)")

        if self.cur_rect:
            self.canvas.delete(self.cur_rect)
            self.cur_rect = None

        self.update_canvas()

    def save_current_image(self):
        if self.img is None:
            messagebox.showwarning("주의", "먼저 이미지를 여세요.")
            return

        path = filedialog.asksaveasfilename(
            title="이미지 저장",
            defaultextension=".png",
            filetypes=[
                ("PNG", "*.png"),
                ("JPEG", "*.jpg;*.jpeg"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return

        save_image(self.img, path)

    # 버튼 기능
    def do_flip_h(self):
        if self.img is None:
            return
        self.push_history()
        self.img = flip_horizontal(self.img)
        self.update_canvas()

    def do_flip_v(self):
        if self.img is None:
            return
        self.push_history()
        self.img = flip_vertical(self.img)
        self.update_canvas()

    def do_blur_all(self):
        if self.img is None:
            return
        self.push_history()
        self.img = blur(self.img, ksize=5)
        self.update_canvas()

    def enable_drag_crop(self):
        """드래그로 자르기 모드 활성화"""
        if self.img is None:
            messagebox.showwarning("주의", "먼저 이미지를 여세요.")
            return
        self.mode = "crop"
        self.mode_label.config(text="현재 모드: Crop (드래그한 영역만 남기고 잘라냄)")
        messagebox.showinfo("Crop 모드", "이미지 위에서 드래그하면, 그 영역만 남기고 잘라냅니다.\n(한 번 자른 뒤에는 다시 Blur 모드로 돌아갑니다.)")

    # 마우스 이벤트
    def on_mouse_down(self, event):
        if self.img is None:
            return

        self.start_x = event.x
        self.start_y = event.y

        if self.cur_rect:
            self.canvas.delete(self.cur_rect)
            self.cur_rect = None

        self.cur_rect = self.canvas.create_rectangle(
            self.start_x, self.start_y, self.start_x, self.start_y,
            outline="red", width=2
        )

    def on_mouse_drag(self, event):
        if self.img is None or not self.cur_rect:
            return

        self.canvas.coords(self.cur_rect, self.start_x, self.start_y, event.x, event.y)

    def on_mouse_up(self, event):
        if self.img is None or not self.cur_rect:
            return

        end_x, end_y = event.x, event.y

        x1 = min(self.start_x, end_x)
        y1 = min(self.start_y, end_y)
        x2 = max(self.start_x, end_x)
        y2 = max(self.start_y, end_y)

        # 드래그하고 드래그 박스 삭제코드
        self.canvas.delete(self.cur_rect)
        self.cur_rect = None

        # 너무 작은 영역은 무시하도록..
        if x2 - x1 < 3 or y2 - y1 < 3:
            return

        # 이미지 범위로 클램프
        img_w, img_h = self.img.size
        x1 = max(0, min(x1, img_w))
        x2 = max(0, min(x2, img_w))
        y1 = max(0, min(y1, img_h))
        y2 = max(0, min(y2, img_h))

        box = (x1, y1, x2, y2)

        # 동작 나누기
        if self.mode == "crop":
            # 드래그한 크기 그대로 잘라서 새 이미지로
            self.push_history()
            self.img = crop_image(self.img, box)
            self.update_canvas()

            # 한 번 잘랐으면 다시 Blur 모드로 복귀
            self.mode = "blur"
            self.mode_label.config(text="현재 모드: Blur (드래그 시 해당 영역만 블러)")
            return

        # 기본 모드: 드래그 영역 Blur
        region = self.img.crop(box)
        blurred = blur(region, ksize=5)

        self.push_history()
        self.img.paste(blurred, box)
        self.update_canvas()


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageToolApp(root)
    root.mainloop()
