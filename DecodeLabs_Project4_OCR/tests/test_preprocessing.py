import unittest
from pathlib import Path
import tempfile
import cv2
import numpy as np

from main import preprocess_image


class PreprocessingTests(unittest.TestCase):
    def test_preprocessing_returns_grayscale_binary_image(self):
        with tempfile.TemporaryDirectory() as folder:
            image_path = Path(folder) / "input.png"
            image = np.full((120, 240, 3), 255, dtype=np.uint8)
            cv2.putText(image, "HELLO OCR", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
            self.assertTrue(cv2.imwrite(str(image_path), image))

            original, processed = preprocess_image(str(image_path), "adaptive")
            self.assertEqual(original.shape, image.shape)
            self.assertEqual(len(processed.shape), 2)
            self.assertEqual(set(np.unique(processed)).issubset({0, 255}), True)

    def test_missing_file_raises_error(self):
        with self.assertRaises(FileNotFoundError):
            preprocess_image("this-file-does-not-exist.png")


if __name__ == "__main__":
    unittest.main()
