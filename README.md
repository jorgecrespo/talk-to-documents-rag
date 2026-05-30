# talk-to-documents-rag

RAG local para consultar PDFs con Chroma, embeddings locales y OpenAI para la generación final.

## Qué hace

- carga PDFs desde `data/raw/`
- extrae texto por página
- divide el texto en chunks
- persiste los chunks en Chroma dentro de `data/chroma/`
- recupera contexto relevante por similitud
- responde preguntas con OpenAI usando ese contexto
- expone una UI simple en Streamlit

## Stack

- Python
- Streamlit
- Chroma
- `sentence-transformers`
- `BAAI/bge-small-en-v1.5`
- OpenAI API
- `pypdf`

## Flujo

1. `loader.py` lee los PDFs.
2. `chunking.py` divide el texto en fragmentos.
3. `indexer.py` genera embeddings y guarda todo en Chroma.
4. `retriever.py` busca los chunks más relevantes.
5. `qa_chain.py` arma el prompt y consulta a OpenAI.
6. `streamlit_app.py` conecta la carga, el índice y las preguntas.

## Estructura

```txt
talk-to-documents-rag/
├── app/
│   ├── ingestion/
│   ├── retrieval/
│   ├── rag/
│   └── ui/
├── data/
│   ├── raw/
│   ├── processed/
│   └── chroma/
├── README.md
├── requirements.txt
├── .env.example
└── .gitignore
```

## Cómo correrlo

### 1. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 2. Configurar OpenAI

```bash
export OPENAI_API_KEY="tu_api_key"
```

### 3. Indexar documentos

```bash
python -m app.ingestion.indexer
```

### 4. Probar retrieval

```bash
python -m app.retrieval.retriever "tu pregunta"
```

### 5. Probar QA

```bash
python -m app.rag.qa_chain "tu pregunta"
```

### 6. Abrir la UI

```bash
streamlit run streamlit_app.py
```

## Notas

- `data/processed/` guarda salidas intermedias para inspección.
- `data/chroma/` contiene el índice persistente.
- La UI permite subir PDFs, indexarlos y hacer preguntas sobre ellos.
