import logging
import os

# import click
import torch
from llama_index.core import Document
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.embeddings.instructor import InstructorEmbedding

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.core.extractors import TitleExtractor
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
import sys 

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from src.constants import (
    EMBEDDING_MODEL_NAME,
    EMBEDDING_TYPE,
    PERSIST_DIRECTORY,
    SOURCE_DIRECTORY,
    cfg
)



def load_documents(source_dir: str) -> list[Document]:
    return SimpleDirectoryReader(source_dir).load_data()

def main(device_type="cpu"):
    chroma_client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
    chroma_collection = chroma_client.get_or_create_collection("chroma_store")

    if EMBEDDING_TYPE == "ollama":
        embed_model = OllamaEmbedding(model_name=EMBEDDING_MODEL_NAME)
    elif EMBEDDING_TYPE == "hf":
        embed_model = InstructorEmbedding(model_name=EMBEDDING_MODEL_NAME, cache_folder="./models", device=device_type)
    else:
        raise NotImplementedError()
    # Load documents and split in chunks
    logging.info(f"Loading documents from {SOURCE_DIRECTORY}")
    documents = load_documents(SOURCE_DIRECTORY)    

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context, embed_model=embed_model, show_progress=True
    )
    


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO
    )
    main(cfg.MODEL.DEVICE)