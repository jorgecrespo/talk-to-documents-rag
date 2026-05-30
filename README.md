# talk-to-documents-rag

Base del proyecto para procesar PDFs localmente.

## Día 1

- leer PDFs desde `data/raw/`
- extraer texto por página
- partir texto en chunks
- guardar resultados intermedios en `data/processed/`

## Día 2

- instalar dependencias con `pip install -r requirements.txt`
- poner PDFs en `data/raw/`
- ejecutar `python -m app.ingestion.indexer`
- revisar el índice persistido en `data/chroma/`

## Día 3

- consultar el índice con `python -m app.retrieval.retriever "tu pregunta"`
- revisar los chunks devueltos con su archivo y página de origen
