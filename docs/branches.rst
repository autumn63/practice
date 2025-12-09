Branches
========

This project uses multiple branches within a single repository to organize
different functionalities.  
Each branch processes a different type of data.

-------------------------
Text Branch
-------------------------

- **Branch name:** ``text``  
- **Purpose:** Detects and removes profanity in text input.

**Pipeline**
    - Input → Normalize → Detect → Clean → Output

**Modules**
    - ``profanity_filter.py`` — core filtering engine  
    - ``badwords.py`` — profanity dataset  
    - ``utils.py`` — normalization utilities  
    - ``file.py`` — saving logs and results  

**Technology**
    - Python, regex, unicodedata  

**Effect**
    - Produces clean, safe text output.


-------------------------
Image Branch
-------------------------

- **Branch name:** ``image``  
- **Purpose:** Load → blur → flip → crop images.

**Modules**
    - ``blur.py`` — Gaussian blur  
    - ``flip_horizontal.py`` / ``flip_vertical.py`` — mirroring  
    - ``crop.py`` — ROI cropping  
    - ``load_image.py`` / ``save_image.py`` — I/O utilities  
    - ``main.py`` — pipeline controller  

**Tech**
    - Python, PIL  

**Effect**
    - Fast preprocessing pipeline for image datasets.


-------------------------
Audio Branch
-------------------------

- **Branch name:** ``audio``  
- **Purpose:** Extract audio from video and remove silent sections.

**Pipeline**
    - Input → Convert → Visualize → Process → Output  

**Modules**
    - ``main.py`` — controller  
    - ``convert.py`` — MP4 → WAV extraction  
    - ``process.py`` — silence detection (Librosa)  
    - ``file.py`` — file saving  

**Tech**
    - MoviePy, Librosa, NumPy  


-------------------------
Video Branch
-------------------------

There are two main functionalities within the video branch:

**1) Video Standardization (main1.py)**  
- Resize to 16:9  
- Apply CLAHE  
- Save sampled frames  
- Generate sequential numpy dataset  

**2) Video Rendering (main2.py)**  
- Load frames  
- Rebuild as MP4 using XVID codec  

**3) Face Blur Filter (main3.py)**  
- MediaPipe-based face detection  
- Expand bounding box  
- Apply strong Gaussian blur  
- Output processed video  

**Tech**
    - OpenCV, MediaPipe, NumPy  


-------------------------
Main Branch
-------------------------

- **Branch Name:** ``main``  
- **Role:** Collects and consolidates final code for the project.
