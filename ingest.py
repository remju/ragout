import os
from PIL import Image
import pytesseract
from pdf2image import convert_from_path

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Paramètres
DOCS_FOLDER = "docs"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# Initialiser embeddings et vectorstore
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)


def process_pdf(pdf_path):
    """Traite un PDF scanné page par page et indexe dans Chroma"""
    print(f"Processing: {pdf_path}")
    
    images = convert_from_path(pdf_path)
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    
    for i, image in enumerate(images):
        # OCR page
        text = pytesseract.image_to_string(image, lang="eng+fra+nld").strip()
        
        if not text:
            print(f"  Page {i+1} is empty, skipping")
            continue  # skip empty pages
        
        # Créer document par page
        doc = Document(page_content=text, metadata={"source": pdf_path, "page": i + 1})
        
        # Split en chunks
        chunks = splitter.split_documents([doc])
        
        if not chunks:
            print(f"  Page {i+1} produced no chunks, skipping")
            continue
        
        # Ajouter dans Chroma
        vectorstore.add_documents(chunks)
        print(f"  Indexed page {i+1}/{len(images)}")
    

def main():
    for file in os.listdir(DOCS_FOLDER):
        if file.lower().endswith(".pdf"):
            process_pdf(os.path.join(DOCS_FOLDER, file))
    
    print("All PDFs processed!")


if __name__ == "__main__":
    main()