"""
HALO - Stage 1: Ingestion
Splits a source document into chunks and a summary into claims.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
import re


def load_document(path: str) -> str:
    """Load text from a PDF or TXT file."""
    if path.endswith(".pdf"):
        reader = PdfReader(path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def chunk_document(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    """Split the source document into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " "],
    )
    return splitter.split_text(text)


def split_claims(summary: str) -> list[str]:
    """Split an AI-generated summary into individual claims (sentences)."""
    sentences = re.split(r"(?<=[.!?])\s+", summary.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 20]


if __name__ == "__main__":
    contract = """This Agreement is made between Acme Corp (the "Supplier")
    and Beta Ltd (the "Buyer"). The Supplier shall deliver 500 units monthly.
    Payment is due within 30 days of invoice. Either party may terminate
    with 60 days written notice. Late payments incur 2% monthly interest."""

    summary = """Acme Corp will deliver 500 units every month to Beta Ltd.
    Payment must be made within 15 days. The contract renews automatically
    every year. Either party can terminate with 60 days notice."""

    chunks = chunk_document(contract)
    claims = split_claims(summary)

    print(f"Document chunks: {len(chunks)}")
    print(f"Summary claims: {len(claims)}\n")
    for i, c in enumerate(claims, 1):
        print(f"Claim {i}: {c}")