from sentence_transformers import CrossEncoder
from langchain_core.documents import Document
import config

_RERANKER_MODEL = None

def get_reranker() -> CrossEncoder:
    global _RERANKER_MODEL
    if _RERANKER_MODEL is None:
        print(f" [Re Ranker] Loading {config.RERANKER_MODEL} on {config.RERANKER_DEVICE} ")
        _RERANKER_MODEL = CrossEncoder(config.RERANKER_MODEL, device= config.RERANKER_DEVICE, ) #Loading the re-ranker model

    return _RERANKER_MODEL

def rerank(query: str, documents: list[Document], top_n: int = None) -> list[Document]:

    if not documents:
        return []

    n = top_n or config.FINAL_TOP_K
    model = get_reranker()

    pairs = [[query, doc.page_content] for doc in documents]
    scores = model.predict(pairs)

    scored_docs = []
    for doc, score in zip(documents, scores):
        doc.metadata["rerank_score"] = float(score)
        scored_docs.append(doc)

    scored_docs.sort(key= lambda d: d.metadata["rerank_score"], reverse= True)

    return scored_docs[:n]