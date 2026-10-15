# DecodeLabs Project 4 — Image & Text Recognition (OCR)

A beginner-friendly Python OCR pipeline built around **OpenCV** image preprocessing and **Tesseract OCR**. It converts an image into machine-readable text, reports the average word-confidence score, and checks that score against a configurable threshold (80% by default).

## Features

- Reads PNG, JPG/JPEG, and other formats supported by OpenCV.
- Converts images to grayscale and reduces small-scale noise with Gaussian blur.
- Supports adaptive thresholding, Otsu thresholding, or no thresholding.
- Uses Tesseract OCR with page segmentation mode 6 (a uniform block of text).
- Reports extracted text and average confidence from recognized words.
- Saves a text report and a preprocessed image under `outputs/`.
- Handles missing image paths and missing Tesseract installation with readable errors.

## Project structure

```text
DecodeLabs_Project4_OCR/
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
├── samples/
│   └── sample_text.png
└── tests/
    └── test_preprocessing.py
```

## Requirements

- Python 3.10 or newer
- Tesseract OCR application installed on your computer
- Python packages in `requirements.txt`

### Install Tesseract

- **Windows:** Install the Tesseract OCR Windows build, then add its installation folder to `PATH`. If it is not detected, set the executable path in `main.py` before calling OCR, for example:
  `pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"`
- **Ubuntu/Debian:** `sudo apt-get update && sudo apt-get install tesseract-ocr`
- **macOS (Homebrew):** `brew install tesseract`

Check installation with `tesseract --version`.

## Setup

```bash
python -m venv .venv
```

Activate the environment:

- Windows PowerShell: `.venv\Scripts\Activate.ps1`
- Windows Command Prompt: `.venv\Scripts\activate.bat`
- macOS/Linux: `source .venv/bin/activate`

Install Python dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run the project

From the project root:

```bash
python main.py samples/sample_text.png
```

Choose a threshold method:

```bash
python main.py samples/sample_text.png --threshold otsu
python main.py samples/sample_text.png --threshold none
```

Change the confidence cutoff or output folder:

```bash
python main.py samples/sample_text.png --confidence 80 --output-dir outputs
```

The program prints extracted text and the average confidence. It writes:
- `outputs/sample_text_ocr.txt` — input name, confidence, validation status, and recognized text.
- `outputs/sample_text_preprocessed.png` — image supplied to OCR after preprocessing.

## How confidence validation works

Tesseract returns a confidence score for each recognized word. The script averages the valid word scores and compares the result to the configured minimum (80% by default). This is a practical acceptance check, **not a guarantee of 80% character-level accuracy**. OCR confidence may differ from actual correctness, and results depend on image quality, font, language, and layout.

## Troubleshooting

- **Tesseract not found:** Install the Tesseract application separately from the Python package and ensure it is on `PATH`.
- **No text detected:** Try a sharper image, crop to the text region, improve lighting, or switch `--threshold otsu` / `--threshold none`.
- **Poor output for a rotated page:** Rotate/crop the image before running OCR.
- **Another language:** Install that language's Tesseract trained data and pass its code with `--language`, e.g. `--language hin` if Hindi trained data is installed.

## Tests

Install dependencies first, then run:

```bash
python -m unittest discover -s tests -v
```

The test checks image loading and preprocessing. It does not require Tesseract to perform OCR.

## GitHub upload

1. Create a new repository on GitHub, for example `DecodeLabs-Project-4-OCR`.
2. Extract this ZIP and open a terminal in the extracted project folder.
3. Run:

```bash
git init
git add .
git commit -m "Complete DecodeLabs Project 4 OCR pipeline"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/DecodeLabs-Project-4-OCR.git
git push -u origin main
```

Replace `YOUR-USERNAME` with your GitHub username. Do not upload passwords, access tokens, or private information.

## Project objective

Demonstrate a basic image-to-text recognition workflow using available AI/computer-vision libraries, preprocessing, confidence reporting, and a clear output. This implementation follows the OCR path from the DecodeLabs Project 4 brief.
