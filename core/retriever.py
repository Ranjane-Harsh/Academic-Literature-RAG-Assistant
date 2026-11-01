from langchain_core.documents import Document
from core.vector_store import similarity_search
from core.reranker import rerank
import config

def retrieve(query: str, top_k: int = None) -> list[Document]:

    final_k = top_k or config.FINAL_TOP_K

    candidates = similarity_search(query=query, top_k=config.INITIAL_TOP_K)
    if not candidates:
        return []

    reranked_docs = rerank(query=query, documents=candidates, top_n=final_k)

    return reranked_docs