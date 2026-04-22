from __future__ import annotations

import json
from pathlib import Path

from google import genai
from google.genai import types

from config import GEMINI_API_KEY

INPUT_DIR = Path("chunked pdf")
OUTPUT_DIR = Path("embeddings")
MODEL = "gemini-embedding-2-preview"


class Embedder:
    def __init__(self, model: str = MODEL) -> None:
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.model = model

    def embed_pdf(self, pdf_path: Path) -> list[float]:
        pdf_bytes = pdf_path.read_bytes()
        result = self.client.models.embed_content(
            model=self.model,
            contents=[
                types.Part.from_bytes(
                    data=pdf_bytes,
                    mime_type="application/pdf",
                ),
            ],
        )
        return list(result.embeddings[0].values)

    def embed_directory(
        self, input_dir: Path = INPUT_DIR, output_dir: Path = OUTPUT_DIR
    ) -> dict[str, list[float]]:
        output_dir.mkdir(parents=True, exist_ok=True)
        results: dict[str, list[float]] = {}
        for pdf in sorted(input_dir.glob("*.pdf")):
            embedding = self.embed_pdf(pdf)
            results[pdf.name] = embedding
            out = output_dir / f"{pdf.stem}.json"
            out.write_text(
                json.dumps({"file": pdf.name, "embedding": embedding})
            )
            print(f"{pdf.name}: dim={len(embedding)} -> {out}")
        return results


def main() -> None:
    Embedder().embed_directory()


if __name__ == "__main__":
    main()
