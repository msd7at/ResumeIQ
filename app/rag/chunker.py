import re
from dataclasses import dataclass, field


# Known section headings — matched case-insensitively at line start
_SECTION_PATTERNS = [
    (re.compile(r"^\s*(summary|objective|profile|about me)\s*$", re.IGNORECASE),    "SUMMARY"),
    (re.compile(r"^\s*(skills?|technologies|tech stack|tools|competencies)\s*$", re.IGNORECASE), "SKILLS"),
    (re.compile(r"^\s*(experience|work history|employment|career)\s*$", re.IGNORECASE), "EXPERIENCE"),
    (re.compile(r"^\s*(education|academic|degree|qualification)\s*$", re.IGNORECASE), "EDUCATION"),
    (re.compile(r"^\s*(projects?|portfolio|personal projects?)\s*$", re.IGNORECASE), "PROJECTS"),
    (re.compile(r"^\s*(certifications?|certificates?|courses?)\s*$", re.IGNORECASE), "CERTIFICATIONS"),
    (re.compile(r"^\s*(awards?|achievements?|honours?|honors?)\s*$", re.IGNORECASE), "ACHIEVEMENTS"),
    (re.compile(r"^\s*(languages?)\s*$", re.IGNORECASE),                             "LANGUAGES"),
    (re.compile(r"^\s*(contact|personal info|personal details)\s*$", re.IGNORECASE), "CONTACT"),
]

MAX_CHUNK_CHARS = 800   # soft cap per chunk — keeps embeddings focused
OVERLAP_LINES   = 2     # lines of overlap between adjacent chunks in same section


@dataclass
class Chunk:
    section: str          # e.g. "EXPERIENCE", "SKILLS", "HEADER"
    text: str             # chunk content
    chunk_index: int      # position within the resume (0-based)
    metadata: dict = field(default_factory=dict)


def chunk_resume(text: str, session_id: str = "") -> list[Chunk]:
    """
    Split resume text into semantically meaningful chunks.
    Each chunk belongs to a section (SKILLS, EXPERIENCE, etc.).
    Long sections are split further at MAX_CHUNK_CHARS with OVERLAP_LINES overlap.
    """
    lines = text.splitlines()
    sections: list[tuple[str, list[str]]] = []  # [(section_name, [lines])]
    current_section = "HEADER"
    current_lines: list[str] = []

    for line in lines:
        matched_section = _detect_section(line)
        if matched_section:
            if current_lines:
                sections.append((current_section, current_lines))
            current_section = matched_section
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_section, current_lines))

    chunks: list[Chunk] = []
    idx = 0

    for section_name, section_lines in sections:
        section_chunks = _split_section(section_name, section_lines)
        for chunk_text in section_chunks:
            if chunk_text.strip():
                chunks.append(Chunk(
                    section=section_name,
                    text=chunk_text.strip(),
                    chunk_index=idx,
                    metadata={"session_id": session_id, "section": section_name},
                ))
                idx += 1

    return chunks


def _detect_section(line: str) -> str | None:
    """Return section label if this line is a section heading, else None."""
    for pattern, label in _SECTION_PATTERNS:
        if pattern.match(line):
            return label
    return None


def _split_section(section: str, lines: list[str]) -> list[str]:
    """
    Split a section's lines into chunks <= MAX_CHUNK_CHARS.
    Adds OVERLAP_LINES from the previous chunk for context continuity.
    """
    chunks = []
    current: list[str] = []
    overlap_buffer: list[str] = []

    for line in lines:
        current.append(line)
        current_text = "\n".join(current)

        if len(current_text) >= MAX_CHUNK_CHARS:
            chunk_text = f"[{section}]\n" + current_text
            chunks.append(chunk_text)
            # keep last OVERLAP_LINES as context for next chunk
            overlap_buffer = current[-OVERLAP_LINES:] if len(current) >= OVERLAP_LINES else current[:]
            current = overlap_buffer[:]

    if current:
        chunk_text = f"[{section}]\n" + "\n".join(current)
        chunks.append(chunk_text)

    return chunks


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python chunker.py <plain_text_file>")
        sys.exit(1)

    with open(sys.argv[1], encoding="utf-8") as f:
        raw = f.read()

    result = chunk_resume(raw, session_id="test-session")
    for c in result:
        print(f"\n── Chunk {c.chunk_index} [{c.section}] ({len(c.text)} chars) ──")
        print(c.text[:300])
    print(f"\nTotal chunks: {len(result)}")
