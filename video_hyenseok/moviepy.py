#moviepy를 활용. 이미지 여러개를 이어서 영상 생성.
#pip install moviepy 필요하다.

from moviepy.editor import *

clip = ImagesequenceClip(['frame1.png', 'frame2.png', 'frame3.png'], fps=1)

clip.ipython_display(width=360)
