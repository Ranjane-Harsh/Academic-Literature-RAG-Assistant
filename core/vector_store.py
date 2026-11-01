from langchain_chroma import Chroma
from langchain_core.documents import Document
from core.embeddings import get_embedding_model
import config
import os

def get_store() -> Chroma:
    return Chroma(
        persist_directory=config.CHROMA_DIR,
        embedding_function=get_embedding_model(),
    )

def add_documents(chunks: list[Document]) -> None:
    store = get_store()

    ids = []
    for i, doc in enumerate(chunks):
        source_name = os.path.basename(doc.metadata.get("source", "doc"))
        page_num = doc.metadata.get("page", 0)
        chunk_id = f"{source_name}_p{page_num}_c{i}"
        ids.append(chunk_id)

    store.add_documents(documents=chunks, ids = ids)
    print(f" [Vector Store] Indexed {len(chunks)} with deterministic IDs")

def similarity_search(query: str, top_k : int = None) -> list[Document]:
    k = top_k or config.INITIAL_TOP_K  # Searching for top 15 candidates for retrival
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
