import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import config

def load_pdf(filepath: str) -> list[Document]:
    doc = pymupdf.open(filepath)
    pages = []

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text()

        if not text.strip():
            continue

        pages.append(Document(page_content = text, metadata = {"source": filepath, "page": page_num}))

    doc.close()

    return pages

def chunk_documents(pages: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size = config.CHUNK_SIZE, chunk_overlap = config.CHUNK_OVERLAP)

    return splitter.split_documents(pages)

def ingest_pdf(filepath: str) -> list[Document]:
    pages = load_pdf(filepath)
    chunks = chunk_documents(pages)
    return chunks