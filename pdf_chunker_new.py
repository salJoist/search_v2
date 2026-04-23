from __future__ import annotations

import json
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

import fitz

CONFIG_PATH = Path(__file__).parent / "sections.json"


@dataclass
class SectionRule:
    name: str
    keywords: list[str]
    contentIndicators: list[str] = field(default_factory=list)


@dataclass
class Section:
    startPage: int
    title: str


@dataclass
class TextBlock:
    pageNumber: int
    text: str
    top: float
    fontSize: float


def loadConfig(configPath: Path = CONFIG_PATH) -> tuple[list[SectionRule], float, int]:
    raw = json.loads(configPath.read_text())
    rules = [
        SectionRule(
            name=entry["name"],
            keywords=entry["keywords"],
            contentIndicators=entry.get("contentIndicators", []),
        )
        for entry in raw["sections"]
    ]
    fuzzyThreshold = raw.get("fuzzyThreshold", 0.6)
    contentMatchThreshold = raw.get("contentMatchThreshold", 2)
    return rules, fuzzyThreshold, contentMatchThreshold


class SectionClassifier:
    def __init__(
        self,
        rules: list[SectionRule],
        fuzzyThreshold: float,
        contentMatchThreshold: int,
    ) -> None:
        self.rules = rules
        self.fuzzyThreshold = fuzzyThreshold
        self.contentMatchThreshold = contentMatchThreshold

    def classifyTitle(self, title: str) -> str | None:
        lowerTitle = title.lower()

        for rule in self.rules:
            for keyword in rule.keywords:
                if keyword in lowerTitle:
                    return rule.name

        for rule in self.rules:
            ratio = SequenceMatcher(None, lowerTitle, rule.name.lower()).ratio()
            if ratio >= self.fuzzyThreshold:
                return rule.name

        return None

    def classifyContent(self, page: fitz.Page) -> str | None:
        pageText = page.get_text().lower()

        for rule in self.rules:
            if not rule.contentIndicators:
                continue
            hits = sum(1 for w in rule.contentIndicators if w in pageText)
            if hits >= self.contentMatchThreshold:
                return rule.name

        return None


class SectionDetector:
    def __init__(self, classifier: SectionClassifier) -> None:
        self.classifier = classifier

    def detectSections(self, pdfPath: Path) -> list[Section]:
        sections = [Section(startPage=1, title="Introduction")]

        with fitz.open(pdfPath) as document:
            for pageIndex, page in enumerate(document):
                pageNumber = pageIndex + 1
                textBlocks = self.extractTextBlocks(page)

                sectionName = self.resolveSectionName(page, textBlocks)
                if not sectionName:
                    continue

                if pageNumber == 1:
                    sections[0] = Section(startPage=1, title=sectionName)
                elif sectionName != sections[-1].title:
                    sections.append(Section(startPage=pageNumber, title=sectionName))

        return sections

    def resolveSectionName(
        self,
        page: fitz.Page,
        textBlocks: list[TextBlock],
    ) -> str | None:
        title = self.selectSectionTitle(textBlocks)
        if title:
            matched = self.classifier.classifyTitle(title)
            if matched:
                return matched

        return self.classifier.classifyContent(page)

    def extractTextBlocks(self, page: fitz.Page) -> list[TextBlock]:
        blocks: list[TextBlock] = []

        for block in page.get_text("dict")["blocks"]:
            if block.get("type") != 0:
                continue

            lines = block.get("lines", [])
            textParts: list[str] = []
            fontSizes: list[float] = []

            for line in lines:
                for span in line.get("spans", []):
                    text = span["text"].strip()
                    if text:
                        textParts.append(text)
                        fontSizes.append(span["size"])

            text = " ".join(textParts).strip()
            if not text:
                continue

            blocks.append(
                TextBlock(
                    pageNumber=page.number + 1,
                    text=text,
                    top=block["bbox"][1],
                    fontSize=max(fontSizes) if fontSizes else 0.0,
                )
            )

        return sorted(blocks, key=lambda b: b.top)

    def selectSectionTitle(self, textBlocks: list[TextBlock]) -> str | None:
        candidates: list[tuple[float, str]] = []

        for block in textBlocks:
            if len(block.text.split()) > 10 or len(block.text) > 80 or block.text.endswith("."):
                continue

            score = block.fontSize

            if block.top < 250:
                score += 5

            candidates.append((score, block.text))

        if not candidates:
            return None

        return max(candidates, key=lambda item: item[0])[1]


class PdfSectionWriter:
    def writeSections(self, pdfPath: Path, outputDir: Path, sections: list[Section]) -> list[Path]:
        outputDir.mkdir(parents=True, exist_ok=True)
        writtenFiles: list[Path] = []

        with fitz.open(pdfPath) as document:
            for index, section in enumerate(sections):
                startPage = section.startPage
                endPage = (
                    sections[index + 1].startPage - 1
                    if index + 1 < len(sections)
                    else document.page_count
                )
                fileName = f"{pdfPath.stem}__{index + 1:02d}_{section.title.replace(' ', '_')}.pdf"
                outputPath = outputDir / fileName

                with fitz.open() as chunk:
                    chunk.insert_pdf(document, from_page=startPage - 1, to_page=endPage - 1)
                    chunk.save(outputPath)

                writtenFiles.append(outputPath)

        return writtenFiles


OUTPUT_DIR = Path("chunks")


def main() -> None:
    import sys

    if len(sys.argv) < 2:
        print("Usage: python pdf_chunker_new.py <path-to-pdf>")
        sys.exit(1)

    pdfPath = Path(sys.argv[1])
    if not pdfPath.is_file():
        print(f"File not found: {pdfPath}")
        sys.exit(1)

    outputDir = OUTPUT_DIR / pdfPath.stem
    rules, fuzzyThreshold, contentMatchThreshold = loadConfig()
    classifier = SectionClassifier(rules, fuzzyThreshold, contentMatchThreshold)
    detector = SectionDetector(classifier)
    writer = PdfSectionWriter()

    sections = detector.detectSections(pdfPath)
    writtenFiles = writer.writeSections(pdfPath, outputDir, sections)
    print(f"{pdfPath.name}: {len(writtenFiles)} section(s) -> {outputDir}/")


if __name__ == "__main__":
    main()
