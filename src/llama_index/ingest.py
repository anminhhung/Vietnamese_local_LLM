import logging
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

# import click
import torch
from llama_index.core import Document
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.core.ingestion import IngestionPipeline, IngestionCache
from llama_index.core.extractors import TitleExtractor
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
import sys 

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from src.constants import (
    EMBEDDING_MODEL_NAME,
    PERSIST_DIRECTORY,
    SOURCE_DIRECTORY,
    cfg
)



def load_documents(source_dir: str) -> list[Document]:
    return SimpleDirectoryReader(source_dir).load_data()

def main(device_type="cpu"):
    chroma_client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
    chroma_collection = chroma_client.get_or_create_collection("quickstart")
    embed_model = OllamaEmbedding(model_name="e5-mistral")

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