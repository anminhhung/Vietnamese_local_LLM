import logging
import os
# import click
import torch
from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceInstructEmbeddings, OllamaEmbeddings

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.document_loaders import NotionDBLoader
import sys 

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.constants import (
    CHROMA_SETTINGS,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_TYPE,
    PERSIST_DIRECTORY,
    SOURCE_DIRECTORY,
    cfg
)

text_template = """
Title: {title}

Tags: {tag}

{content}

"""

def file_log(logentry):
   file1 = open(cfg.STORAGE.INGEST_LOG,"a")
   file1.write(logentry + "\n")
   file1.close()
   print(logentry + "\n")

def load_notion_db() -> Document:
    # Loads a single document from a file path


    try:
        loader = NotionDBLoader(
            integration_token="secret_KvnxRdvdB1ft9d4e30nQRN60lY5yinSEPwP5JIsf39i",
            database_id="ebd082cd324b42238927e92443f791a9"
        )
        docs = loader.load()
        processed_docs = []

        for doc in docs:
            processed_text = text_template.format(title=doc.metadata['name'], tag="; ".join(doc.metadata['tags']), content=doc.page_content)
            processed_docs.append(Document(page_content=processed_text, metadata={"status": "; ".join(doc.metadata['status'])}))
        return processed_docs
    
    except Exception as ex:
       file_log('loading error \n')
       return None 


def main(device_type="cpu"):
    # Load documents and split in chunks
    documents = load_notion_db()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)

    logging.info(f"Loaded {len(documents)} documents from {SOURCE_DIRECTORY}")
    logging.info(f"Split into {len(texts)} chunks of text")

    # Create embeddings
    if EMBEDDING_TYPE == "hf":
        embeddings = HuggingFaceInstructEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            cache_folder = "./models",
            model_kwargs={"device": device_type},
        )
    elif EMBEDDING_TYPE == "ollama":
        embeddings = OllamaEmbeddings(
            model=EMBEDDING_MODEL_NAME
        )
    else:
        raise NotImplementedError

    # change the embedding type here if you are running into issues.
    # These are much smaller embeddings and will work for most appications
    # If you use HuggingFaceEmbeddings, make sure to also use the same in the
    # run_localGPT.py file.

    # embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    db = Chroma.from_documents(
        texts,
        embeddings,
        persist_directory=PERSIST_DIRECTORY,
        client_settings=CHROMA_SETTINGS,
    )
   


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO
    )
    main(cfg.MODEL.DEVICE)