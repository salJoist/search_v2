from dataclasses import dataclass
from pathlib import Path

import fitz
import torch
from doclayout_yolo import YOLOv10
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer, util


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


REPO_ID = "juliozhao/DocLayout-YOLO-DocStructBench"
MODEL_FILE = "doclayout_yolo_docstructbench_imgsz1024.pt"


class DocLayoutModel:
    def __init__(self, repoId: str = REPO_ID, fileName: str = MODEL_FILE) -> None:
        modelPath = hf_hub_download(repo_id=repoId, filename=fileName)
        self.model = YOLOv10(modelPath)

    def detectTitleRegions(self, pageImagePath: Path) -> list[tuple[float, float, float, float]]:
        result = self.model.predict(str(pageImagePath), imgsz=1024, conf=0.2, verbose=False)[0]
        names = result.names
        titleBoxes: list[tuple[float, float, float, float]] = []

        for box in result.boxes:
            classId = int(box.cls[0].item())
            label = names[classId].lower()
            if "title" in label or "header" in label:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                titleBoxes.append((x1, y1, x2, y2))

        return sorted(titleBoxes, key=lambda b: b[1])


SECTION_DESCRIPTIONS: dict[str, str] = {
    "Cover Page": "cover page with project title company name and logo",
    "Transmittal Letter": "letter of transmittal or cover letter addressed to the client",
    "Table Of Contents": "table of contents listing sections and page numbers",
    "Organizational Chart": "organizational chart showing team structure and reporting hierarchy",
    "Project Experience": "project experience past performance relevant projects case studies with client budget scope",
    "Project Approach": "project approach technical methodology and design strategy",
    "Quality Control": "quality control quality assurance QA QC procedures and standards",
    "Key Personnel": "key personnel team members staff resumes qualifications education certifications registration",
    "Scope Of Work": "scope of work scope of services deliverables and tasks",
    "Pricing": "pricing cost proposal fee schedule budget compensation and rates",
    "Schedule": "project schedule timeline milestones and phasing plan",
    "Safety": "safety plan health and safety program OSHA compliance",
    "Conclusion": "conclusion closing statement and final summary",
    "Introduction": "introduction executive summary and project overview",
}

RESUME_INDICATORS = [
    "education", "registration", "certifications", "professional registration",
    "years of experience", "professional experience", "license",
]

PROJECT_INDICATORS = [
    "client:", "owner:", "completion date", "contract value",
    "services provided", "reference:", "project cost", "budget:",
]

CLASSIFIER_MODEL = "all-MiniLM-L6-v2"
TITLE_THRESHOLD = 0.4


class SectionClassifier:
    def __init__(self, modelName: str = CLASSIFIER_MODEL) -> None:
        self.model = SentenceTransformer(modelName)
        self.sectionLabels = list(SECTION_DESCRIPTIONS.keys())
        descriptions = list(SECTION_DESCRIPTIONS.values())
        self.descriptionEmbeddings = self.model.encode(descriptions, convert_to_tensor=True)

    def classifyTitle(self, title: str) -> str | None:
        titleEmbedding = self.model.encode(title, convert_to_tensor=True)
        scores = util.cos_sim(titleEmbedding, self.descriptionEmbeddings)[0]
        bestIdx = torch.argmax(scores).item()
        bestScore = scores[bestIdx].item()
        if bestScore >= TITLE_THRESHOLD:
            return self.sectionLabels[bestIdx]
        return None

    @staticmethod
    def classifyContent(page: fitz.Page) -> str | None:
        pageText = page.get_text().lower()
        resumeHits = sum(1 for w in RESUME_INDICATORS if w in pageText)
        projectHits = sum(1 for w in PROJECT_INDICATORS if w in pageText)

        if resumeHits >= 2:
            return "Key Personnel"
        if projectHits >= 2:
            return "Project Experience"
        return None


class SectionDetector:
    def __init__(self, layoutModel: DocLayoutModel, classifier: SectionClassifier) -> None:
        self.layoutModel = layoutModel
        self.classifier = classifier

    def detectSections(self, pdfPath: Path) -> list[Section]:
        sections = [Section(startPage=1, title="Introduction")]

        with fitz.open(pdfPath) as document:
            for pageIndex, page in enumerate(document):
                pageNumber = pageIndex + 1
                imagePath = pdfPath.parent / f"{pdfPath.stem}_page_{pageNumber}.png"
                pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                pixmap.save(imagePath)

                titleRegions = self.layoutModel.detectTitleRegions(imagePath)
                textBlocks = self.extractTextBlocks(page)
                imagePath.unlink(missing_ok=True)

                sectionName = self.resolveSectionName(page, textBlocks, titleRegions)
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
        titleRegions: list[tuple[float, float, float, float]],
    ) -> str | None:
        title = self.selectSectionTitle(textBlocks, titleRegions)
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

    def selectSectionTitle(
        self,
        textBlocks: list[TextBlock],
        titleRegions: list[tuple[float, float, float, float]],
    ) -> str | None:
        candidates: list[tuple[float, str]] = []

        for block in textBlocks:
            if len(block.text.split()) > 10 or len(block.text) > 80 or block.text.endswith("."):
                continue

            score = block.fontSize

            if block.top < 250:
                score += 5

            for _, y1, _, y2 in titleRegions:
                if y1 - 30 <= block.top <= y2 + 30:
                    score += 10
                    break

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
        print("Usage: python embedding_new.py <path-to-pdf>")
        sys.exit(1)

    pdfPath = Path(sys.argv[1])
    if not pdfPath.is_file():
        print(f"File not found: {pdfPath}")
        sys.exit(1)

    outputDir = OUTPUT_DIR / pdfPath.stem
    layoutModel = DocLayoutModel()
    classifier = SectionClassifier()
    detector = SectionDetector(layoutModel, classifier)
    writer = PdfSectionWriter()

    sections = detector.detectSections(pdfPath)
    writtenFiles = writer.writeSections(pdfPath, outputDir, sections)
    print(f"{pdfPath.name}: {len(writtenFiles)} section(s) -> {outputDir}/")


if __name__ == "__main__":
    main()
