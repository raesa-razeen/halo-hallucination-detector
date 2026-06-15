"""
HALO - Stage 1: Ingestion v0.2
Upgraded from naive character chunking to legal clause segmentation.
Splits documents by actual clause structure - numbered clauses,
headings, and legal section markers.
"""

import re
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_document(path: str) -> str:
    """Load text from a PDF or TXT file."""
    if path.endswith(".pdf"):
        from pypdf import PdfReader
        reader = PdfReader(path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def segment_legal_clauses(text: str) -> list[str]:
    """
    Segment a legal document into clauses based on structure.
    Detects numbered clauses, lettered sections, and legal headings.
    
    Examples detected:
    - "1. Payment Terms"
    - "2.1 Obligations"  
    - "ARTICLE III - Termination"
    - "Section 4: Liability"
    - "(a) The Supplier shall..."
    """

    # Legal clause patterns - ordered from most specific to least
    clause_patterns = [
        r'\n(?=\d+\.\d+\s+[A-Z])',          # Numbered sub-clauses: 2.1 Term
        r'\n(?=\d+\.\s+[A-Z])',              # Numbered clauses: 1. Payment
        r'\n(?=ARTICLE\s+[IVX\d]+)',         # ARTICLE III
        r'\n(?=Section\s+\d+)',              # Section 4
        r'\n(?=SECTION\s+\d+)',              # SECTION 4
        r'\n(?=Clause\s+\d+)',               # Clause 5
        r'\n(?=[A-Z][A-Z\s]{4,}:)',          # ALL CAPS HEADING:
        r'\n(?=\([a-z]\)\s)',                # (a) lettered items
    ]

    # Combine all patterns
    combined_pattern = '|'.join(clause_patterns)

    # Split on clause boundaries
    segments = re.split(combined_pattern, text)

    # Clean and filter segments
    clauses = []
    for segment in segments:
        cleaned = segment.strip()
        if len(cleaned) > 50:  # Filter out very short fragments
            clauses.append(cleaned)

    # If no clause structure detected fall back to smart chunking
    if len(clauses) <= 1:
        return fallback_chunk(text)

    return clauses


def fallback_chunk(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    """
    Fallback to RecursiveCharacterTextSplitter if no clause
    structure is detected in the document.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " "],
    )
    return splitter.split_text(text)


def chunk_document(text: str) -> list[str]:
    """
    Main entry point for document chunking.
    Tries legal clause segmentation first, falls back to
    character chunking if no structure detected.
    """
    clauses = segment_legal_clauses(text)
    return clauses


def split_claims(summary: str) -> list[str]:
    """Split an AI-generated summary into individual claims."""
    sentences = re.split(r'(?<=[.!?])\s+', summary.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 20]


def classify_clause_importance(clause: str) -> str:
    """
    Classify clause importance based on legal keywords.
    Implements Algorithm 1 from ICCIC-2025 paper - upgraded
    from word count to keyword-based importance detection.

    High importance: payment, termination, liability, warranty,
                     indemnification, confidentiality
    Medium importance: delivery, obligations, representations
    Low importance: boilerplate, definitions, general provisions
    """
    clause_lower = clause.lower()

    high_importance_keywords = [
        'payment', 'terminate', 'termination', 'liability', 'liable',
        'warranty', 'indemnif', 'confidential', 'penalty', 'damages',
        'breach', 'default', 'intellectual property', 'governing law'
    ]

    medium_importance_keywords = [
        'deliver', 'obligation', 'represent', 'warrant', 'comply',
        'notice', 'consent', 'approval', 'assign', 'transfer'
    ]

    for keyword in high_importance_keywords:
        if keyword in clause_lower:
            return "HIGH"

    for keyword in medium_importance_keywords:
        if keyword in clause_lower:
            return "MEDIUM"

    return "LOW"


if __name__ == "__main__":
    # Test with a properly structured legal contract
    contract = """
1. PARTIES
This Agreement is made between Acme Corp (the "Supplier") and Beta Ltd (the "Buyer").

2. DELIVERY OBLIGATIONS
The Supplier shall deliver 500 units monthly to the Buyer's designated warehouse.
Delivery shall occur within the first 5 business days of each month.

3. PAYMENT TERMS
Payment is due within 30 days of invoice. Late payments incur 2% monthly interest.
The Buyer shall not withhold payment for any reason without written notice.

4. TERMINATION
Either party may terminate this Agreement with 60 days written notice.
Immediate termination is permitted in cases of material breach.

5. CONFIDENTIALITY
Both parties agree to maintain strict confidentiality of all shared information.
This obligation survives termination of the Agreement for 3 years.
    """

    summary = """Acme Corp will deliver 500 units every month to Beta Ltd.
    Payment must be made within 15 days. The contract renews automatically
    every year. Either party can terminate with 60 days notice."""

    print("HALO v0.2 - Legal Clause Segmentation\n")
    print("=" * 50)

    clauses = segment_legal_clauses(contract)
    claims = split_claims(summary)

    print(f"Clauses detected: {len(clauses)}")
    print(f"Claims extracted: {len(claims)}\n")

    print("CLAUSE BREAKDOWN WITH IMPORTANCE:")
    print("-" * 50)
    for i, clause in enumerate(clauses, 1):
        importance = classify_clause_importance(clause)
        preview = clause[:80].replace('\n', ' ')
        print(f"Clause {i} [{importance}]: {preview}...")

    print("\nCLAIMS:")
    print("-" * 50)
    for i, claim in enumerate(claims, 1):
        print(f"Claim {i}: {claim}")