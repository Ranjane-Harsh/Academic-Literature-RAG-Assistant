from langchain_chroma import Chroma
from langchain_core.documents import Document
from core.embeddings import get_embedding_model
import config

def get_store() -> Chroma:
    return Chroma(
        persist_directory=config.CHROMA_DIR,
        embedding_function=get_embedding_model(),
    )

def add_documents(chunks: list[Document]) -> None:
    store = get_store()
    store.add_documents(chunks)

def similarity_search(query: str, top_k = None) -> list[Document]:
    k = top_k or config.TOP_K
    store = get_store()
    instruction = "Represent this sentence for searching relevant passages: "
    formatted_query = f"{instruction}{query}"
    return store.similarity_search(formatted_query, k=k)

def list_papers() -> list[str]:

    store = get_store()
    result = store.get()
    sources = [m["source"] for m in result["metadatas"] if m and "source" in m ]

    return sorted(set(sources))

def delete_paper(filepath: str) -> None:
    store = get_store()
    result = store.get(where={"source": filepath})
    ids_to_delete = result["ids"]

    if ids_to_delete:
        store.delete(ids=ids_to_delete)
        print(f"Deleted {len(ids_to_delete)} chunks for {filepath}")

    else:
        print(f"No chunks found for {filepath}")
