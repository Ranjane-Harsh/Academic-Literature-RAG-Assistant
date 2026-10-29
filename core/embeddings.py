from langchain_huggingface import HuggingFaceEmbeddings
import config

def get_embedding_model() -> HuggingFaceEmbeddings:
    
    return HuggingFaceEmbeddings(
        model_name = config.EMBEDDING_MODEL,
        model_kwargs = {"device": config.EMBEDDING_DEVICE},
        encode_kwargs = {"normalize_embeddings": True},
    )