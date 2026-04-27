import re
from dataclasses import dataclass, field


# Regex patterns for contact info detection
_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
_PHONE_RE = re.compile(r"(\+?\d[\d\s\-().]{7,}\d)")

# Section heading keywords — checked case-insensitively
_SECTION_KEYWORDS = {
    "skills":     ["skill", "technologies", "tech stack", "tools", "competencies"],
    "experience": ["experience", "employment", "work history", "career"],
    "education":  ["education", "degree", "university", "college", "academic"],
    "projects":   ["project", "portfolio"],
    "summary":    ["summary", "objective", "profile", "about"],
}


@dataclass
class ValidationResult:
    is_valid: bool
    missing_fields: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def validate_resume(text: str) -> ValidationResult:
    """
    Check a resume text for missing critical fields.
    Returns a ValidationResult with is_valid flag, missing_fields, and warnings.
    """
    if not text or not text.strip():
        return ValidationResult(is_valid=False, missing_fields=["entire resume content"])

    missing = []
    warnings = []
    lower = text.lower()

    # ── Contact info ────────────────────────────────────────
    if not _EMAIL_RE.search(text):
        missing.append("email address")

    if not _PHONE_RE.search(text):
        missing.append("phone number")

    # ── Core sections ────────────────────────────────────────
    for section, keywords in _SECTION_KEYWORDS.items():
        found = any(kw in lower for kw in keywords)
        if not found:
            if section in ("skills", "experience", "education"):
                missing.append(f"{section} section")
            else:
                warnings.append(f"{section} section not found (optional but recommended)")

    # ── Length sanity check ──────────────────────────────────
    word_count = len(text.split())
    if word_count < 50:
        missing.append("sufficient content (resume appears too short)")
    elif word_count < 150:
        warnings.append("resume seems short — consider adding more detail")

    is_valid = len(missing) == 0
    return ValidationResult(is_valid=is_valid, missing_fields=missing, warnings=warnings)


if __name__ == "__main__":
    sample = """
    John Doe  |  john.doe@email.com  |  +91-9876543210

    SUMMARY
    Backend developer with 5 years of experience in Java and Python.

    EXPERIENCE
    Senior Developer — Infosys (2020–2024)
    - Built REST APIs with Spring Boot and FastAPI.

    EDUCATION
    B.Tech Computer Science — VIT University, 2019

    SKILLS
    Java, Python, FastAPI, Docker, PostgreSQL
    """

    result = validate_resume(sample)
    print(f"Valid: {result.is_valid}")
    print(f"Missing: {result.missing_fields}")
    print(f"Warnings: {result.warnings}")
