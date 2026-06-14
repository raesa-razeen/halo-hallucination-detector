"""
HALO - Hallucination Audit Layer for Official Documents
Streamlit UI - v0.1
"""

import streamlit as st
from verifier import build_vector_store, retrieve_evidence, verify_claim, chunk_document, split_claims
from scorer import calculate_fidelity_score

# Page config
st.set_page_config(
    page_title="HALO - Hallucination Detector",
    page_icon="",
    layout="wide"
)

# Header
st.title("HALO")
st.subheader("Hallucination Audit Layer for Official Documents")
st.markdown("*Implements the Legal Fidelity Score from published ICCIC-2025 research*")
st.divider()

# Input columns
col1, col2 = st.columns(2)

with col1:
    st.markdown("### Original Document")
    document_text = st.text_area(
        "Paste the original legal contract or document here",
        height=300,
        placeholder="Paste your original document here..."
    )

with col2:
    st.markdown("### AI Generated Summary")
    summary_text = st.text_area(
        "Paste the AI-generated summary to verify",
        height=300,
        placeholder="Paste the AI-generated summary here..."
    )

st.divider()

# Run button
run_button = st.button("Run HALO Analysis", type="primary", use_container_width=True)

if run_button:
    if not document_text.strip() or not summary_text.strip():
        st.error("Please provide both a document and a summary before running analysis.")
    else:
        with st.spinner("Running hallucination detection..."):

            # Stage 1 - Ingestion
            chunks = chunk_document(document_text)
            claims = split_claims(summary_text)

            # Stage 2 - Build vector store and verify
            collection = build_vector_store(chunks)
            results = []
            for claim in claims:
                evidence = retrieve_evidence(claim, collection)
                result = verify_claim(claim, evidence)
                results.append(result)

            # Stage 3 - Score
            score_data = calculate_fidelity_score(results)

        st.divider()
        st.markdown("## HALO Audit Report")

        # Score metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Legal Fidelity Score", f"{score_data['fidelity_score']}%")
        with col2:
            st.metric("Hallucination Rate", f"{score_data['hallucination_rate']}%")
        with col3:
            st.metric("Total Claims", score_data['total_claims'])
        with col4:
            st.metric("Hallucinated", score_data['hallucinated_count'])

        st.divider()
        st.markdown("### Claim-Level Breakdown")

        # Per claim results
        for i, r in enumerate(results, 1):
            verdict = r.get("verdict", "N/A")

            if verdict == "SUPPORTED":
                color = "green"
                label = "SUPPORTED"
            elif verdict == "NOT_SUPPORTED":
                color = "red"
                label = "NOT SUPPORTED"
            else:
                color = "orange"
                label = "PARTIAL"

            with st.expander(f"Claim {i}: {r.get('claim', '')} — :{color}[{label}]"):
                st.markdown(f"**Verdict:** :{color}[{label}]")
                st.markdown(f"**Reason:** {r.get('reason', 'N/A')}")
                st.markdown(f"**Evidence from document:** {r.get('evidence', 'N/A')}")

        # Hallucinated claims summary
        if score_data["hallucinated_claims"]:
            st.divider()
            st.markdown("### Hallucinated Claims Flagged")
            st.error("The following claims were not supported by the original document:")
            for c in score_data["hallucinated_claims"]:
                st.markdown(f"- {c}")

        # Supported claims summary
        if score_data["supported_claims"]:
            st.divider()
            st.markdown("### Verified Claims")
            st.success("The following claims were fully supported by the original document:")
            for c in score_data["supported_claims"]:
                st.markdown(f"- {c}")