"""
HALO - Stage 3: Legal Fidelity Scorer
Implements the Legal Fidelity Score from the published ICCIC-2025 research.
Converts claim-level verdicts into a weighted document-level score.
"""


def calculate_fidelity_score(results: list[dict]) -> dict:
    """
    Calculate the Legal Fidelity Score from verification results.

    Scoring logic:
    - SUPPORTED           = 1.0 (fully grounded)
    - PARTIALLY_SUPPORTED = 0.5 (partially grounded)
    - NOT_SUPPORTED       = 0.0 (hallucinated)

    Returns overall score, per-claim breakdown, and hallucination summary.
    """
    if not results:
        return {"error": "No results to score"}

    verdict_scores = {
        "SUPPORTED": 1.0,
        "PARTIALLY_SUPPORTED": 0.5,
        "NOT_SUPPORTED": 0.0
    }

    total_score = 0.0
    hallucinated = []
    supported = []
    partial = []

    for r in results:
        verdict = r.get("verdict", "NOT_SUPPORTED").strip()
        score = verdict_scores.get(verdict, 0.0)
        total_score += score

        if verdict == "NOT_SUPPORTED":
            hallucinated.append(r.get("claim", ""))
        elif verdict == "SUPPORTED":
            supported.append(r.get("claim", ""))
        elif verdict == "PARTIALLY_SUPPORTED":
            partial.append(r.get("claim", ""))

    fidelity_score = (total_score / len(results)) * 100
    hallucination_rate = (len(hallucinated) / len(results)) * 100

    return {
        "fidelity_score": round(fidelity_score, 2),
        "hallucination_rate": round(hallucination_rate, 2),
        "total_claims": len(results),
        "supported_count": len(supported),
        "partial_count": len(partial),
        "hallucinated_count": len(hallucinated),
        "hallucinated_claims": hallucinated,
        "supported_claims": supported,
        "partial_claims": partial
    }


def print_audit_report(results: list[dict], score_data: dict):
    """Print a clean audit report to terminal."""
    print("\n" + "=" * 60)
    print("         HALO AUDIT REPORT")
    print("=" * 60)

    print(f"\nLEGAL FIDELITY SCORE: {score_data['fidelity_score']}%")
    print(f"HALLUCINATION RATE:   {score_data['hallucination_rate']}%")
    print(f"TOTAL CLAIMS:         {score_data['total_claims']}")
    print(f"SUPPORTED:            {score_data['supported_count']}")
    print(f"PARTIALLY SUPPORTED:  {score_data['partial_count']}")
    print(f"HALLUCINATED:         {score_data['hallucinated_count']}")

    print("\n" + "-" * 60)
    print("CLAIM-LEVEL BREAKDOWN:")
    print("-" * 60)

    for i, r in enumerate(results, 1):
        verdict = r.get("verdict", "N/A")
        label = "[SUPPORTED]" if verdict == "SUPPORTED" else "[NOT SUPPORTED]" if verdict == "NOT_SUPPORTED" else "[PARTIAL]"
        print(f"\n{label} Claim {i}: {r.get('claim', '')}")
        print(f"   Verdict:  {verdict}")
        print(f"   Reason:   {r.get('reason', 'N/A')}")
        print(f"   Evidence: {r.get('evidence', 'N/A')}")

    if score_data["hallucinated_claims"]:
        print("\n" + "-" * 60)
        print("HALLUCINATED CLAIMS FLAGGED:")
        print("-" * 60)
        for c in score_data["hallucinated_claims"]:
            print(f"  - {c}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    from verifier import (
        build_vector_store, retrieve_evidence,
        verify_claim, chunk_document, split_claims
    )

    contract = """This Agreement is made between Acme Corp (the "Supplier")
    and Beta Ltd (the "Buyer"). The Supplier shall deliver 500 units monthly.
    Payment is due within 30 days of invoice. Either party may terminate
    with 60 days written notice. Late payments incur 2% monthly interest."""

    summary = """Acme Corp will deliver 500 units every month to Beta Ltd.
    Payment must be made within 15 days. The contract renews automatically
    every year. Either party can terminate with 60 days notice."""

    print("HALO - Running Full Analysis...\n")

    chunks = chunk_document(contract)
    claims = split_claims(summary)
    collection = build_vector_store(chunks)

    results = []
    for claim in claims:
        evidence = retrieve_evidence(claim, collection)
        result = verify_claim(claim, evidence)
        results.append(result)

    score_data = calculate_fidelity_score(results)
    print_audit_report(results, score_data)