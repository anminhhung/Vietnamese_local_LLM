import logging
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

# import click
import torch
from llama_index.core import Document
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex
from llama_index.core.ingestion import IngestionPipeline, IngestionCache
from llama_index.core.extractors import TitleExtractor
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.vector_store.qdrant import QdrantVectorStore

from qdrant_client import QdrantClient

import sys 

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.llama_index.constants import (
    CHROMA_SETTINGS,
    EMBEDDING_MODEL_NAME,
    INGEST_THREADS,
    PERSIST_DIRECTORY,
    SOURCE_DIRECTORY,
    cfg
)


client = QdrantClient(":memory:")
vector_store = QdrantVectorStore(client=client, collection_name="default_store")

def load_documents(source_dir: str) -> list[Document]:
    return SimpleDirectoryReader(source_dir).load_data()

def main(device_type="cpu"):
    # Load documents and split in chunks
    logging.info(f"Loading documents from {SOURCE_DIRECTORY}")
    documents = load_documents(SOURCE_DIRECTORY)

    ingestion_pipeline = IngestionPipeline(
        transformations=[
            SemanticSplitterNodeParser()
        ]
    )

    # Create embeddings
    


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO
    )
    main(cfg.MODEL.DEVICE)