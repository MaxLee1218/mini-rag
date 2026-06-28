import re
from pathlib import Path
from typing import Dict, List

from docx import Document
from pypdf import PdfReader


def _clean_text(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = text.strip()
    return text


def _load_txt(file_path: Path) -> str:
    try:
        text = file_path.read_text(encoding="utf-8")
        return _clean_text(text)
    except Exception as e:
        print(f"[Loader Warning] Failed to load {file_path}: {e}")
        return ""


def _load_docx(file_path: Path) -> str:
    try:
        document = Document(file_path)
        paragraphs = [paragraph.text for paragraph in document.paragraphs]
        text = "\n".join(paragraphs)
        return _clean_text(text)
    except Exception as e:
        print(f"[Loader Warning] Failed to load {file_path}: {e}")
        return ""


def _load_pdf(file_path: Path) -> str:
    try:
        reader = PdfReader(file_path)
        page_texts = []

        for page in reader.pages:
            page_text = page.extract_text()
            if page_text is None or page_text.strip() == "":
                continue
            page_texts.append(page_text.strip())

        text = "\n".join(page_texts)
        return _clean_text(text)
    except Exception as e:
        print(f"[Loader Warning] Failed to load {file_path}: {e}")
        return ""


def load_documents(folder_path: str = "data/raw") -> List[Dict[str, str]]:
    base_path = Path(folder_path)
    if not base_path.exists():
        return []

    documents = []
    loaders = {
        ".pdf": _load_pdf,
        ".txt": _load_txt,
        ".docx": _load_docx,
    }

    for file_path in base_path.rglob("*"):
        if not file_path.is_file():
            continue

        loader = loaders.get(file_path.suffix.lower())
        if loader is None:
            continue

        cleaned_text = loader(file_path)
        if cleaned_text:
            source = file_path.relative_to(base_path).as_posix()
            documents.append({"text": cleaned_text, "source": source})

    return documents
