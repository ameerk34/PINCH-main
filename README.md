# PINCH

Simple setup and run steps for this project.

## 1) Requirements

This project uses:

- `numpy==1.26.4`
- `opencv-python==4.10.0.84`
- `pandas==2.2.2` (recommended for logs/plots)
- `matplotlib==3.9.0` (recommended for logs/plots)
- `mediapipe`
- `tensorflow` (or `tflite-runtime==2.14.0` on some Linux/Raspberry Pi setups)
- `ultralytics==8.3.11`
- `torch==2.3.1`
- `torchvision==0.18.1`
- `tqdm==4.66.4`

## 2) Create and activate virtual environment (Windows PowerShell)

```powershell
cd "C:\Users\ameer\OneDrive\Desktop\PINCH-main"
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## 3) Install dependencies

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

## 4) Run the live app

```powershell
cd letsgo
python pinchreader.py
```

## 5) Optional: run experiment script

```powershell
cd ..
python exp1.py --pt {path_to_yolo_pt}
```

Replace `{path_to_yolo_pt}` with your YOLO `.pt` model path.
