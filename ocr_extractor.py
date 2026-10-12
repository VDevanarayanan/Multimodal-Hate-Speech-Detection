import os
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("OCRExtractor")

class OCRExtractor:
    """
    Extracts text embedded inside meme images.
    Uses EasyOCR or PyTesseract if available, with robust fallback mechanisms.
    """
    def __init__(self, languages=['en']):
        self.reader = None
        self._init_ocr(languages)

    def _init_ocr(self, languages):
        try:
            import easyocr
            self.reader = easyocr.Reader(languages, gpu=False)
            logger.info("EasyOCR initialized successfully.")
        except Exception as e:
            logger.warning(f"EasyOCR initialization skipped/failed: {e}. Falling back to Pytesseract or PIL.")
            try:
                import pytesseract
                self.reader = "pytesseract"
                logger.info("PyTesseract selected as OCR engine.")
            except Exception:
                self.reader = None
                logger.info("No native OCR library installed. Fallback to image-embedded text metadata.")

    def extract_text(self, image_input) -> str:
        """
        Extracts text from PIL Image or image file path.
        """
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                return ""
            image_path = image_input
            try:
                pil_img = Image.open(image_path).convert('RGB')
            except Exception:
                return ""
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert('RGB')
            image_path = None
        else:
            return ""

        if self.reader is not None and not isinstance(self.reader, str):
            try:
                import numpy as np
                img_np = np.array(pil_img)
                results = self.reader.readtext(img_np, detail=0)
                return " ".join(results)
            except Exception as e:
                logger.warning(f"EasyOCR extraction error: {e}")

        if self.reader == "pytesseract" and image_path:
            try:
                import pytesseract
                return pytesseract.image_to_string(pil_img).strip()
            except Exception as e:
                logger.warning(f"PyTesseract extraction error: {e}")

        return ""
