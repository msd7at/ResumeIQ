import fitz  # PyMuPDF
from pathlib import Path


def parse_pdf(file_path: str) -> str:
    """
    Extract all text from a PDF file, page by page.
    Returns a single cleaned string of the full resume text.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, got: {path.suffix}")

    doc = fitz.open(file_path)
    pages_text = []

    for page in doc:
        text = page.get_text("text")   # plain text, preserves line breaks
        text = text.strip()
        if text:
            pages_text.append(text)

    doc.close()

    full_text = "\n\n".join(pages_text)
    return _clean_text(full_text)


def _clean_text(text: str) -> str:
    """Remove noise: excessive blank lines and non-printable characters."""
    lines = text.splitlines()
    cleaned = []
    prev_blank = False

    for line in lines:
        line = line.strip()
        # skip non-printable / garbage characters from scanned PDFs
        line = "".join(ch for ch in line if ch.isprintable())
        is_blank = len(line) == 0

        # collapse multiple consecutive blank lines into one
        if is_blank and prev_blank:
            continue
        cleaned.append(line)
        prev_blank = is_blank

    return "\n".join(cleaned).strip()


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python pdf_parser.py <path_to_resume.pdf>")
        sys.exit(1)

    text = parse_pdf(sys.argv[1])
    print("── Extracted Text ──────────────────────")
    print(text[:2000])   # print first 2000 chars for preview
    print(f"\n── Total characters extracted: {len(text)}")
