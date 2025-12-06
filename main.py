# main.py
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from PIL import ImageTk

from src.load_image import load_image
from src.save_image import save_image
from src.blur import blur
from src.resize import resize
from src.flip_horizontal import flip_horizontal
from src.flip_vertical import flip_vertical


class ImageToolApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Tool Test - blur / resize / flip")

        # 현재 이미지
        self.img = None
        self.original_img = None      # 원본 백업
        self.tk_img = None
        self.image_id = None

        # Undo로 이전 상태 저장
        self.history = []             # 리스트에 저장된 이미지들을 쌓아둠

        # 드래그 선택 사각형으로..
        self.start_x = None
        self.start_y = None
        self.cur_rect = None

        # 상단 버튼들 구현
        top_frame = tk.Frame(self.root)
        top_frame.pack(side=tk.TOP, fill=tk.X)

        open_btn = tk.Button(top_frame, text="이미지 열기", command=self.open_image)
        open_btn.pack(side=tk.LEFT, padx=5, pady=5)

        save_btn = tk.Button(top_frame, text="이미지 저장", command=self.save_current_image)
        save_btn.pack(side=tk.LEFT, padx=5, pady=5)

        # 기능 버튼 영역
        func_frame = tk.Frame(self.root)
        func_frame.pack(side=tk.TOP, fill=tk.X)

        flip_h_btn = tk.Button(func_frame, text="좌우 반전", command=self.apply_flip_horizontal)
        flip_h_btn.pack(side=tk.LEFT, padx=5, pady=5)

        flip_v_btn = tk.Button(func_frame, text="상하 반전", command=self.apply_flip_vertical)
        flip_v_btn.pack(side=tk.LEFT, padx=5, pady=5)

        blur_all_btn = tk.Button(func_frame, text="전체 Blur", command=self.apply_blur_all)
        blur_all_btn.pack(side=tk.LEFT, padx=5, pady=5)

        resize_btn = tk.Button(func_frame, text="리사이즈", command=self.apply_resize)
        resize_btn.pack(side=tk.LEFT, padx=5, pady=5)

        undo_btn = tk.Button(func_frame, text="되돌리기 (Undo)", command=self.undo)
        undo_btn.pack(side=tk.LEFT, padx=5, pady=5)

        info_label = tk.Label(
            func_frame,
            text="이미지 드래그: 선택 영역 Blur / 버튼: flip, resize, 전체 blur, 되돌리기"
        )
        info_label.pack(side=tk.LEFT, padx=10)

        # 이미지 전처리 할 캔버스
        self.canvas = tk.Canvas(self.root, bg="gray")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # 영역선택을 위한 코드
        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_move_press)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)

    # 캔버스 이미지 갱신
    def update_canvas_image(self):
        if self.img is None:
            return

        w, h = self.img.size
        self.canvas.config(width=w, height=h)

        self.tk_img = ImageTk.PhotoImage(self.img)
        if self.image_id is None:
            self.image_id = self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)
        else:
            self.canvas.itemconfig(self.image_id, image=self.tk_img)

    # Undo 히스토리 관리
    def push_history(self):
        """이미지를 변경하기 직전 상태를 history에 저장"""
        if self.img is not None:
            self.history.append(self.img.copy())

    def undo(self):
        """이전 이미지 상태로 되돌리기"""
        if not self.history:
            messagebox.showinfo("Undo", "되돌릴 작업이 없습니다.")
            return

        self.img = self.history.pop()
        self.update_canvas_image()

    # 파일 열기 + 저장
    def open_image(self):
        file_path = filedialog.askopenfilename(
            title="이미지 파일 선택",
            filetypes=[
                ("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif"),
                ("All files", "*.*")
            ]
        )
        if not file_path:
            return

        try:
            self.img = load_image(file_path)  # load_image.py 사용
            self.original_img = self.img.copy()
        except FileNotFoundError as e:
            messagebox.showerror("에러", str(e))
            return

        # 새 이미지 열면 히스토리 초기화 되게 함
        self.history.clear()

        # 선택 사각형 초기화 시킴
        if self.cur_rect is not None:
            self.canvas.delete(self.cur_rect)
            self.cur_rect = None

        self.update_canvas_image()

    def save_current_image(self):
        if self.img is None:
            messagebox.showwarning("주의", "먼저 이미지를 여세요.")
            return

        file_path = filedialog.asksaveasfilename(
            title="이미지 저장",
            defaultextension=".png",
            filetypes=[
                ("PNG", "*.png"),
                ("JPEG", "*.jpg;*.jpeg"),
                ("BMP", "*.bmp"),
                ("All files", "*.*")
            ]
        )
        if not file_path:
            return

        try:
            save_image(self.img, file_path)  # save_image.py 사용
            messagebox.showinfo("완료", f"이미지가 저장되었습니다.\n{file_path}")
        except Exception as e:
            messagebox.showerror("에러", f"저장 중 오류 발생: {e}")

    # 버튼: flip / blur / resize 테스트
    def apply_flip_horizontal(self):
        if self.img is None:
            messagebox.showwarning("주의", "먼저 이미지를 여세요.")
            return
        self.push_history()                 # 변경 전 상태 저장
        self.img = flip_horizontal(self.img)
        self.update_canvas_image()

    def apply_flip_vertical(self):
        if self.img is None:
            messagebox.showwarning("주의", "먼저 이미지를 여세요.")
            return
        self.push_history()
        self.img = flip_vertical(self.img)
        self.update_canvas_image()

    def apply_blur_all(self):
        if self.img is None:
            messagebox.showwarning("주의", "먼저 이미지를 여세요.")
            return
        self.push_history()
        self.img = blur(self.img, ksize=5)
        self.update_canvas_image()

    def apply_resize(self):
        if self.img is None:
            messagebox.showwarning("주의", "먼저 이미지를 여세요.")
            return

        new_w = simpledialog.askinteger("리사이즈", "새 가로 크기(px):", minvalue=1)
        if new_w is None:
            return
        new_h = simpledialog.askinteger("리사이즈", "새 세로 크기(px):", minvalue=1)
        if new_h is None:
            return

        keep = messagebox.askyesno("비율 유지", "가로세로 비율을 유지할까요? (Yes: keep_ratio=True)")

        self.push_history()
        self.img = resize(self.img, new_w, new_h, keep_ratio=keep)
        self.update_canvas_image()

    #  드래그로 선택 영역 Blur 
    def on_button_press(self, event):
        if self.img is None:
            return

        self.start_x = event.x
        self.start_y = event.y

        # 기존 사각형 삭제
        if self.cur_rect is not None:
            self.canvas.delete(self.cur_rect)
            self.cur_rect = None

        # 새 사각형 생성
        self.cur_rect = self.canvas.create_rectangle(
            self.start_x, self.start_y, self.start_x, self.start_y,
            outline="red", width=2
        )

    def on_move_press(self, event):
        if self.img is None or self.cur_rect is None:
            return

        cur_x, cur_y = event.x, event.y
        self.canvas.coords(self.cur_rect, self.start_x, self.start_y, cur_x, cur_y)

    def on_button_release(self, event):
        if self.img is None or self.cur_rect is None:
            return

        end_x, end_y = event.x, event.y

        # 좌표 정렬
        x1 = min(self.start_x, end_x)
        y1 = min(self.start_y, end_y)
        x2 = max(self.start_x, end_x)
        y2 = max(self.start_y, end_y)

        img_w, img_h = self.img.size
        x1 = max(0, min(x1, img_w))
        x2 = max(0, min(x2, img_w))
        y1 = max(0, min(y1, img_h))
        y2 = max(0, min(y2, img_h))

        # 너무 작은 선택은 무시하도록
        if x2 - x1 < 3 or y2 - y1 < 3:
            self.canvas.delete(self.cur_rect)
            self.cur_rect = None
            return

        # 해당 영역만 crop → blur → paste
        box = (x1, y1, x2, y2)
        region = self.img.crop(box)
        blurred_region = blur(region, ksize=5)

        # 드래그 blur도 변경 전 상태를 저장
        self.push_history()
        self.img.paste(blurred_region, box)

        self.update_canvas_image()

        # 선택 사각형 제거
        self.canvas.delete(self.cur_rect)
        self.cur_rect = None


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageToolApp(root)
    root.mainloop()
