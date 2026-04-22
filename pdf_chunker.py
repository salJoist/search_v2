from __future__ import annotations

import re
from pathlib import Path

import pymupdf
from google import genai
from google.genai import types
from pydantic import BaseModel

from config import GEMINI_API_KEY

INPUT_DIR = Path("pdfs")
OUTPUT_DIR = Path("chunked pdf")
MODEL = "gemini-2.5-flash"
PROMPT = (
    "Split this proposal PDF into its MAJOR top-level sections, the way a table "
    "of contents would list them. Group related content under a single section: "
    "multiple project case studies belong to ONE 'Project Experience' section, "
    "multiple team-member bios belong to ONE 'Key Personnel' section, and so on. "
    "Do NOT create a separate section per page or per sub-heading. Most proposals "
    "have between 5 and 10 top-level sections.\n\n"
    "Return a JSON array of {start_page, title}. Entries must be ordered by "
    "start_page, non-overlapping, and cover every page; the first entry must "
    "start at page 1. Titles should be concise (<=6 words)."
)


class Section(BaseModel):
    start_page: int
    title: str


def _slug(title: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", title).strip("-")[:60] or "section"


def detect_sections(pdf_path: Path, client: genai.Client) -> list[Section]:
    pdf_part = types.Part.from_bytes(
        data=pdf_path.read_bytes(), mime_type="application/pdf"
    )
    response = client.models.generate_content(
        model=MODEL,
        contents=[PROMPT, pdf_part],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=list[Section],
            temperature=0,
        ),
    )
    return sorted(response.parsed, key=lambda s: s.start_page)


def chunk_pdf(pdf_path: Path, output_dir: Path, client: genai.Client) -> list[Path]:
    sections = detect_sections(pdf_path, client)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    with pymupdf.open(pdf_path) as doc:
        for i, section in enumerate(sections):
            start = section.start_page
            end = sections[i + 1].start_page - 1 if i + 1 < len(sections) else doc.page_count
            out = output_dir / (
                f"{pdf_path.stem}__{i + 1:02d}_p{start:03d}-p{end:03d}_{_slug(section.title)}.pdf"
            )
            with pymupdf.open() as mini:
                mini.insert_pdf(doc, from_page=start - 1, to_page=end - 1)
                mini.save(out)
            written.append(out)
    return written


def main() -> None:
    client = genai.Client(api_key=GEMINI_API_KEY)
    for pdf in sorted(INPUT_DIR.glob("*.pdf")):
        paths = chunk_pdf(pdf, OUTPUT_DIR, client)
        print(f"{pdf.name}: {len(paths)} section(s) -> {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()