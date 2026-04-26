import re
from dataclasses import dataclass, field


# Known section headings — matched case-insensitively, full-line match only.
# Each tuple: (compiled regex, section label)
# Multi-word variants listed explicitly so "Professional Experience" etc. are caught.
_SECTION_PATTERNS = [
    (re.compile(
        r"^\s*(summary|professional summary|career summary|executive summary|"
        r"objective|career objective|profile|professional profile|"
        r"about me|personal statement)\s*$", re.IGNORECASE
    ), "SUMMARY"),

    (re.compile(
        r"^\s*(skills?|technical skills?|core skills?|key skills?|professional skills?|"
        r"technologies|tech stack|tools|competencies|expertise|"
        r"areas? of expertise|skillset|skill set)\s*$", re.IGNORECASE
    ), "SKILLS"),

    (re.compile(
        r"^\s*(experience|professional experience|work experience|relevant experience|"
        r"industry experience|internship experience|internships?|"
        r"employment|employment history|work history|career|career history|"
        r"work background|professional background)\s*$", re.IGNORECASE
    ), "EXPERIENCE"),

    (re.compile(
        r"^\s*(education|educational background|academic background|academic|"
        r"academic qualifications?|educational qualifications?|"
        r"degree|qualifications?|schooling)\s*$", re.IGNORECASE
    ), "EDUCATION"),

    (re.compile(
        r"^\s*(projects?|key projects?|notable projects?|personal projects?|"
        r"side projects?|academic projects?|open source|portfolio)\s*$", re.IGNORECASE
    ), "PROJECTS"),

    (re.compile(
        r"^\s*(certifications?|professional certifications?|certificates?|"
        r"licenses?|accreditations?|credentials?|courses?|training)\s*$", re.IGNORECASE
    ), "CERTIFICATIONS"),

    (re.compile(
        r"^\s*(awards?|achievements?|honours?|honors?|recognition|accolades?)\s*$",
        re.IGNORECASE
    ), "ACHIEVEMENTS"),

    (re.compile(
        r"^\s*(languages?|language skills?|languages? known|foreign languages?)\s*$",
        re.IGNORECASE
    ), "LANGUAGES"),

    (re.compile(
        r"^\s*(contact|contact information|contact details?|"
        r"personal info|personal information|personal details?)\s*$", re.IGNORECASE
    ), "CONTACT"),

    (re.compile(
        r"^\s*(publications?|research papers?|papers?|journal articles?|presentations?)\s*$",
        re.IGNORECASE
    ), "PUBLICATIONS"),

    (re.compile(
        r"^\s*(research|research experience|research projects?)\s*$", re.IGNORECASE
    ), "RESEARCH"),

    (re.compile(
        r"^\s*(volunteer|volunteer experience|volunteer work|"
        r"community service|community involvement)\s*$", re.IGNORECASE
    ), "VOLUNTEER"),

    (re.compile(
        r"^\s*(leadership|leadership experience|campus activities|"
        r"student organizations?)\s*$", re.IGNORECASE
    ), "LEADERSHIP"),

    (re.compile(
        r"^\s*(interests?|hobbies|hobbies and interests?|personal interests?|"
        r"extracurricular|extracurricular activities|activities)\s*$", re.IGNORECASE
    ), "INTERESTS"),

    (re.compile(
        r"^\s*(references?|professional references?)\s*$", re.IGNORECASE
    ), "REFERENCES"),
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
