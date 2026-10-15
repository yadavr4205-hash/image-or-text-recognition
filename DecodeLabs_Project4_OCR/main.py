"""DecodeLabs Project 4: Image-to-text OCR pipeline."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import cv2
import pytesseract
from pytesseract import Output


def preprocess_image(image_path: str, threshold: str = "adaptive"):
    """Load an image and prepare it for OCR. Returns (original, processed)."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(
            f"Could not open image: {image_path}. Check the path and file format."
        )

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.GaussianBlur(gray, (3, 3), 0)

    if threshold == "adaptive":
        processed = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 31, 11
        )
    elif threshold == "otsu":
        _, processed = cv2.threshold(
            denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
    else:
        processed = denoised

    return image, processed


def run_ocr(image_path: str, output_dir: str = "outputs",
            min_confidence: float = 80.0, threshold: str = "adaptive",
            language: str = "eng"):
    """Run OCR and save extracted text plus a preprocessed preview."""
    original, processed = preprocess_image(image_path, threshold)
    output = pytesseract.image_to_data(
        processed, lang=language, output_type=Output.DICT,
        config="--psm 6"
    )

    words, confidences = [], []
    for word, conf in zip(output["text"], output["conf"]):
        word = word.strip()
        try:
            confidence = float(conf)
        except (TypeError, ValueError):
            continue
        if word and confidence >= 0:
            words.append((word, confidence))
            confidences.append(confidence)

    extracted_text = pytesseract.image_to_string(
        processed, lang=language, config="--psm 6"
    ).strip()
    average_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    passed = average_confidence >= min_confidence

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    stem = Path(image_path).stem
    text_path = destination / f"{stem}_ocr.txt"
    image_path_out = destination / f"{stem}_preprocessed.png"
    text_path.write_text(
        f"Input: {image_path}\n"
        f"Average word confidence: {average_confidence:.2f}%\n"
        f"Required threshold: {min_confidence:.2f}%\n"
        f"Validation: {'PASS' if passed else 'BELOW THRESHOLD'}\n\n"
        f"Extracted text:\n{extracted_text or '[No text detected]'}\n",
        encoding="utf-8"
    )
    cv2.imwrite(str(image_path_out), processed)

    return {
        "text": extracted_text,
        "confidence": average_confidence,
        "threshold": min_confidence,
        "passed": passed,
        "text_file": str(text_path),
        "processed_image": str(image_path_out),
        "words": words,
    }


def main():
    parser = argparse.ArgumentParser(
        description="DecodeLabs Project 4 — OCR text extraction from an image."
    )
    parser.add_argument("image", help="Path to an input image (PNG/JPG/JPEG/BMP)")
    parser.add_argument("--output-dir", default="outputs", help="Directory for results")
    parser.add_argument("--confidence", type=float, default=80.0,
                        help="Minimum average OCR confidence (default: 80)")
    parser.add_argument("--threshold", choices=["adaptive", "otsu", "none"],
                        default="adaptive", help="Image thresholding method")
    parser.add_argument("--language", default="eng", help="Installed Tesseract language code")
    args = parser.parse_args()

    if not 0 <= args.confidence <= 100:
        parser.error("--confidence must be between 0 and 100")

    try:
        result = run_ocr(args.image, args.output_dir, args.confidence,
                         args.threshold, args.language)
    except (FileNotFoundError, pytesseract.TesseractNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        if isinstance(exc, pytesseract.TesseractNotFoundError):
            print("Install the Tesseract OCR application; see README.md.", file=sys.stderr)
        raise SystemExit(1)

    print("\n========== DecodeLabs OCR Result ==========")
    print(result["text"] or "[No text detected]")
    print(f"\nAverage word confidence: {result['confidence']:.2f}%")
    print(f"80% validation: {'PASS' if result['passed'] else 'BELOW THRESHOLD'}")
    print(f"Text report: {result['text_file']}")
    print(f"Preprocessed image: {result['processed_image']}")
    if not result["passed"]:
        print("Tip: try a clearer image, better lighting, or --threshold otsu/none.")
    print("===========================================\n")


if __name__ == "__main__":
    main()
