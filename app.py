import io

import streamlit as st

from src.config import (
    GENAI_API_KEY,
    PINECONE_API_KEY,
    PINECONE_INDEX,
    PINECONE_DIMENSION,
    PINECONE_METRIC,
    PINECONE_CLOUD,
    PINECONE_REGION,
)
from src.embedding_service import EmbeddingService
from src.pinecone_client import PineconeClient
from src.contract_recommender import ContractRecommender
from src.ui import render_results_table, render_score_bar


def main() -> None:
    """Entry point for the ClauseRadar Streamlit application."""
    st.set_page_config(
        page_title="ClauseRadar",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.sidebar.title("🔍 ClauseRadar")

    # Instantiate PineconeClient early to fetch index stats
    pinecone_client = PineconeClient(
        api_key=PINECONE_API_KEY,
        cloud=PINECONE_CLOUD,
        region=PINECONE_REGION,
        index_name=PINECONE_INDEX,
        dimension=PINECONE_DIMENSION,
        metric=PINECONE_METRIC,
    )

    # Determine how many vectors currently exist in the index
    stats = pinecone_client.index.describe_index_stats()
    namespaces = stats.get("namespaces", {})
    total_vectors = sum(ns.get("vector_count", 0) for ns in namespaces.values())
    # Ensure max_k is at least 1 so the slider is valid
    max_k = total_vectors if total_vectors > 0 else 1

    # PDF upload section
    st.sidebar.subheader("Index New Contracts")
    uploaded_files = st.sidebar.file_uploader(
        "Upload one or more contract PDFs", type=["pdf"], accept_multiple_files=True
    )

    # Keywords input section
    st.sidebar.markdown("Enter keywords or phrases, separated by commas:")
    keywords_input = st.sidebar.text_input(
        "e.g. Effective Date, Payment Terms, Termination"
    )
    keywords_list = [k.strip() for k in keywords_input.split(",") if k.strip()]

    # Index button
    if st.sidebar.button("⏫ Index Contracts"):
        if not uploaded_files:
            st.sidebar.error("🔺 Please upload at least one PDF before indexing.")
        elif not keywords_list:
            st.sidebar.error("🔺 Please enter at least one keyword or phrase.")
        else:
            with st.spinner("Indexing…"):
                contract_files: dict[str, io.BytesIO] = {}
                for pdf in uploaded_files:
                    pdf_bytes = pdf.read()
                    contract_id = pdf.name.rsplit(".", 1)[0]
                    contract_files[contract_id] = io.BytesIO(pdf_bytes)

                embed_service = EmbeddingService(api_key=GENAI_API_KEY)
                # Reuse the same pinecone_client to avoid reinitializing
                recommender = ContractRecommender(
                    embedding_service=embed_service, pinecone_client=pinecone_client
                )

                recommender.index_contracts(contract_files, keywords_list)

            st.sidebar.success("✅ Indexing complete!")
            st.sidebar.info(
                f"Indexed {len(contract_files)} contract(s) against "
                f"{len(keywords_list)} keyword(s)."
            )

            # Recompute total_vectors after indexing
            stats = pinecone_client.index.describe_index_stats()
            namespaces = stats.get("namespaces", {})
            total_vectors = sum(ns.get("vector_count", 0) for ns in namespaces.values())
            max_k = total_vectors if total_vectors > 0 else 1

    st.sidebar.markdown("---")

    # Search section
    st.sidebar.subheader("Search Contracts")
    search_query = st.sidebar.text_input(
        "Type any keyword or free-text query", placeholder="e.g. Confidential Information"
    )

    # Slider now uses dynamic max based on total_vectors
    top_k = st.sidebar.slider(
        "Number of results to return", 1, max_k, min(5, max_k)
    )

    # Handle search button click: perform embedding + Pinecone, store results
    if st.sidebar.button("🔎 Search"):
        if not search_query.strip():
            st.sidebar.error("🔺 Enter a keyword or phrase before searching.")
        else:
            embed_service = EmbeddingService(api_key=GENAI_API_KEY)
            recommender = ContractRecommender(
                embedding_service=embed_service, pinecone_client=pinecone_client
            )

            # Progress bar and status text
            progress = st.progress(0)
            status_text = st.empty()

            # Step 1: Embed the query
            status_text.text("🛰️ Embedding search query…")
            query_emb = embed_service.embed_text(search_query)
            progress.progress(50)

            # Step 2: Query Pinecone
            status_text.text("🔍 Querying Pinecone for similar clauses…")
            raw_matches = pinecone_client.query(query_emb, top_k=top_k)
            progress.progress(100)

            # Clear progress indicators
            progress.empty()
            status_text.empty()

            # Build a simple list of dicts and store in session_state
            session_results = []
            for match in raw_matches:
                meta = match["metadata"]
                session_results.append(
                    {
                        "contract_id": meta["contract_id"],
                        "keyword": meta["keyword"],
                        "score": match["score"],
                        "snippet": meta["snippet"],
                    }
                )
            st.session_state["results"] = session_results

    # If results exist in session_state, display them
    if "results" in st.session_state:
        results = st.session_state["results"]

        if not results:
            st.warning(
                "No matches found. Try a different keyword or upload/index more contracts."
            )
        else:
            st.markdown(f"## Top {len(results)} matches for “{search_query}”")
            render_results_table(results)
            render_score_bar(results)

            labels = [
                f"{i + 1}: {row['contract_id']} – {row['keyword']}"
                for i, row in enumerate(results)
            ]
            selection = st.selectbox("View full snippet for:", [""] + labels)
            if selection:
                idx = int(selection.split(":")[0]) - 1
                chosen = results[idx]
                st.markdown("**Full Snippet:**")
                st.write(chosen["snippet"])
                st.info(
                    f"🔗 To download the entire PDF for "
                    f"“{chosen['contract_id']},” use your local copy."
                )


if __name__ == "__main__":
    main()
