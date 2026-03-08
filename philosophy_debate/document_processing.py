from __future__ import annotations

import io
import re
from pathlib import Path

from pypdf import PdfReader

from philosophy_debate.models import ExtractedDocument


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def list_pdf_files(corpus_dir: Path) -> list[Path]:
    return sorted(corpus_dir.glob("*.pdf"))


class PDFTextExtractor:
    def __init__(
        self,
        min_direct_text_chars: int = 1200,
        ocr_language: str = "eng",
        tesseract_cmd: str | None = None,
    ) -> None:
        self.min_direct_text_chars = min_direct_text_chars
        self.ocr_language = ocr_language
        self.tesseract_cmd = tesseract_cmd

    def extract(self, pdf_path: Path) -> ExtractedDocument:
        warnings: list[str] = []
        direct_text, page_count, direct_warning = self._extract_with_pypdf(pdf_path)
        if direct_warning:
            warnings.append(direct_warning)

        cleaned_text = normalize_whitespace(direct_text)
        extractor = "pypdf"

        # OCR only kicks in when direct extraction looks too thin to trust.
        if len(cleaned_text) < self.min_direct_text_chars:
            ocr_text, ocr_warning = self._extract_with_ocr(pdf_path)
            if ocr_warning:
                warnings.append(ocr_warning)
            if len(ocr_text) > len(cleaned_text):
                cleaned_text = normalize_whitespace(ocr_text)
                extractor = "ocr"

        if not cleaned_text:
            raise RuntimeError(f"Could not extract readable text from {pdf_path.name}.")

        return ExtractedDocument(
            source_id=pdf_path.stem,
            title=pdf_path.stem.replace("-", " ").replace("_", " "),
            file_path=str(pdf_path),
            text=cleaned_text,
            extractor=extractor,
            page_count=page_count,
            warnings=warnings,
        )

    def _extract_with_pypdf(self, pdf_path: Path) -> tuple[str, int, str | None]:
        try:
            reader = PdfReader(str(pdf_path))
            page_texts = []
            for page in reader.pages:
                page_texts.append(page.extract_text() or "")
            return "\n\n".join(page_texts), len(reader.pages), None
        except Exception as exc:
            return "", 0, f"Direct extraction failed for {pdf_path.name}: {exc}"

    def _extract_with_ocr(self, pdf_path: Path) -> tuple[str, str | None]:
        try:
            import fitz
            import pytesseract
            from PIL import Image
        except Exception:
            return "", f"OCR libraries are unavailable for {pdf_path.name}."

        if self.tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd

        try:
            document = fitz.open(pdf_path)
            page_texts = []
            for page in document:
                pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
                image_bytes = pixmap.tobytes("png")
                with Image.open(io.BytesIO(image_bytes)) as image:
                    page_texts.append(
                        pytesseract.image_to_string(image, lang=self.ocr_language)
                    )
            return "\n\n".join(page_texts), None
        except Exception as exc:
            return "", f"OCR failed for {pdf_path.name}: {exc}"
