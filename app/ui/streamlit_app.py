"""Streamlit UI for the document RAG demo."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from app.ingestion.indexer import index_pdfs
from app.rag.qa_chain import answer_from_query


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
CHROMA_DIR = Path("data/chroma")


def ensure_directories() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)


def save_uploaded_pdfs(uploaded_files) -> list[str]:
    """Persist uploaded PDFs into the raw data directory."""

    saved_paths: list[str] = []
    for uploaded_file in uploaded_files:
        target_path = RAW_DIR / uploaded_file.name
        target_path.write_bytes(uploaded_file.getbuffer())
        saved_paths.append(target_path.name)
    return saved_paths


def render_indexing_tab() -> None:
    st.subheader("Cargar e indexar PDFs")
    uploaded_files = st.file_uploader(
        "Subí uno o más PDFs",
        type=["pdf"],
        accept_multiple_files=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        chunk_size = st.number_input("Chunk size", min_value=200, max_value=4000, value=1000, step=50)
    with col2:
        overlap = st.number_input("Overlap", min_value=0, max_value=1000, value=150, step=25)

    if st.button("Guardar archivos en data/raw") and uploaded_files:
        saved = save_uploaded_pdfs(uploaded_files)
        st.success(f"Guardados: {', '.join(saved)}")

    if st.button("Indexar documentos"):
        try:
            count = index_pdfs(
                raw_dir=RAW_DIR,
                processed_dir=PROCESSED_DIR,
                chroma_dir=CHROMA_DIR,
                chunk_size=int(chunk_size),
                overlap=int(overlap),
            )
        except Exception as exc:  # noqa: BLE001 - surface the failure in the UI
            st.error(str(exc))
        else:
            st.success(f"Indexados {count} chunks en Chroma.")


def render_question_tab() -> None:
    st.subheader("Preguntar sobre los documentos")
    question = st.text_input("Tu pregunta")
    top_k = st.number_input("Top K", min_value=1, max_value=10, value=3, step=1)

    if st.button("Responder"):
        if not question.strip():
            st.warning("Escribí una pregunta primero.")
            return

        try:
            result = answer_from_query(question=question, chroma_dir=str(CHROMA_DIR), top_k=int(top_k))
        except Exception as exc:  # noqa: BLE001 - surface the failure in the UI
            st.error(str(exc))
            return

        st.markdown("### Respuesta")
        st.write(result.answer)

        if result.sources:
            st.markdown("### Fuentes")
            for source in result.sources:
                st.write(f"- {source.source_file} | página {source.page_number} | chunk {source.chunk_index}")


def main() -> None:
    st.set_page_config(page_title="Talk to Documents", page_icon="📄", layout="wide")
    ensure_directories()

    st.title("Talk to Documents")
    st.write("Subí PDFs, indexalos localmente y hacé preguntas sobre su contenido.")

    tab_index, tab_ask = st.tabs(["Cargar e indexar", "Preguntar"])

    with tab_index:
        render_indexing_tab()

    with tab_ask:
        render_question_tab()


if __name__ == "__main__":
    main()
