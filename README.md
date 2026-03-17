# Local RAG with Ollama

This project is a minimal **RAG (Retrieval-Augmented Generation)** pipeline running fully locally.

It supports:

* Local LLM via Ollama
* PDF ingestion (including scanned PDFs with OCR)
* Multilingual documents (EN / FR / NL)

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/remju/ragout
cd ragout
```

---

### 2. Create virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Install system dependencies

```bash
sudo apt install tesseract-ocr
sudo apt install tesseract-ocr-eng tesseract-ocr-fra tesseract-ocr-nld
```

---

### 5. Install Ollama and models

Install Ollama: https://ollama.com

Then pull models:

```bash
ollama pull llama3
ollama pull nomic-embed-text
```

---

## Ingest documents

Place your PDFs in the `docs/` folder.

Run:

```bash
python ingest.py
```

This will:

* Extract text (OCR if needed)
* Split into chunks
* Store embeddings in Chroma

---

## Query your documents

```bash
python query.py
```

Then ask questions in the terminal.

---

## Notes

* Everything runs locally (no API calls)
* First query may be slow (model loading)
* OCR can be slow for large scanned PDFs
* Made by questionning an LLM about RAGs
* Code produced by GPT-5.3

---