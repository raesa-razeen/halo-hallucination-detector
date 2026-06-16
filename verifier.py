"""
HALO - Stage 2: Grounding & Verification
Embeds document chunks, retrieves relevant evidence for each claim,
and uses LLaMA via Groq to judge if the claim is supported.
"""

import os
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer
import chromadb
from ingestion import chunk_document, split_claims

load_dotenv()

# Initialize embedding model (runs locally, no API cost)
EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize Groq client
GROQ_CLIENT = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Initialize ChromaDB (local vector store)
CHROMA_CLIENT = chromadb.Client()


def build_vector_store(chunks: list[str], collection_name: str = "halo_docs"):
    """Embed document chunks and store in ChromaDB."""
    # Fresh collection every run
    try:
        CHROMA_CLIENT.delete_collection(collection_name)
    except:
        pass

    collection = CHROMA_CLIENT.create_collection(collection_name)
    embeddings = EMBEDDER.encode(chunks).tolist()

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )
    print(f"Vector store built with {len(chunks)} chunks")
    return collection


def retrieve_evidence(claim: str, collection, top_k: int = 3) -> list[str]:
    """Find the most relevant document chunks for a given claim."""
    claim_embedding = EMBEDDER.encode([claim]).tolist()
    results = collection.query(
        query_embeddings=claim_embedding,
        n_results=min(top_k, collection.count())
    )
    return results["documents"][0]


def verify_claim(claim: str, evidence_chunks: list[str]) -> dict:
    """Use LLaMA via Groq to judge if the claim is supported by evidence."""
    evidence_text = "\n---\n".join(evidence_chunks)

    prompt = f"""You are a legal document verification expert. 
Your job is to check if a claim made in a summary is supported by the original document.

ORIGINAL DOCUMENT EXCERPTS:
{evidence_text}

CLAIM TO VERIFY:
{claim}

Analyze carefully and respond in exactly this format:
VERDICT: [SUPPORTED / PARTIALLY_SUPPORTED / NOT_SUPPORTED]
REASON: [One sentence explaining why]
EVIDENCE: [The exact phrase from the document that supports or contradicts this claim, or NONE if not found]
"""

    response = GROQ_CLIENT.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )

    raw = response.choices[0].message.content.strip()

    # Parse the structured response
    result = {"claim": claim, "raw_response": raw}
    for line in raw.split("\n"):
        if line.startswith("VERDICT:"):
            result["verdict"] = line.replace("VERDICT:", "").strip()
        elif line.startswith("REASON:"):
            result["reason"] = line.replace("REASON:", "").strip()
        elif line.startswith("EVIDENCE:"):
            result["evidence"] = line.replace("EVIDENCE:", "").strip()

    return result


if __name__ == "__main__":
    # Same test contract and summary from Stage 1
    contract = """This Agreement is made between Acme Corp (the "Supplier")
    and Beta Ltd (the "Buyer"). The Supplier shall deliver 500 units monthly.
    Payment is due within 30 days of invoice. Either party may terminate
    with 60 days written notice. Late payments incur 2% monthly interest."""

    summary = """Acme Corp will deliver 500 units every month to Beta Ltd.
    Payment must be made within 15 days. The contract renews automatically
    every year. Either party can terminate with 60 days notice."""

    print("HALO - Hallucination Detection Running...\n")

    # Stage 1 - Ingestion
    chunks = chunk_document(contract)
    claims = split_claims(summary)

    # Stage 2 - Build vector store
    collection = build_vector_store(chunks)

    # Stage 2 - Verify each claim
    print("\n VERIFICATION RESULTS:\n")
    for i, claim in enumerate(claims, 1):
        evidence = retrieve_evidence(claim, collection)
        result = verify_claim(claim, evidence)

        print(f"Claim {i}: {claim}")
        print(f"Verdict: {result.get('verdict', 'N/A')}")
        print(f"Reason:  {result.get('reason', 'N/A')}")
        print(f"Evidence: {result.get('evidence', 'N/A')}")
        print("-" * 60)