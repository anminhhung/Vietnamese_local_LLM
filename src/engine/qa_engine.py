import chromadb
import logging
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext

from ..llm_serving.model import load_model
from ..core.prompt_template_utils import get_prompt_template
from ..constants import (
    EMBEDDING_MODEL_NAME,
    EMBEDDING_TYPE,
    PERSIST_DIRECTORY,
    MODEL_ID,
    MODEL_BASENAME,
    SERVICE,
    RESPONSE_MODE,
    cfg
)

def create_retrieval_qa_pipeline(self, device_type="cuda", stream=True):
    """
    Initializes and returns a retrieval-based Question Answering (QA) pipeline.

    """

    llm = load_model(cfg, model_id=MODEL_ID, device_type=device_type, service=SERVICE)

    chroma_client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
    chroma_collection = chroma_client.get_or_create_collection("chroma_db")
    
    if EMBEDDING_TYPE == "ollama":
        embed_model = OllamaEmbedding(model_name=EMBEDDING_MODEL_NAME)
    elif EMBEDDING_TYPE == "hf":
        embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL_NAME, cache_folder="./models", device=device_type)
    elif  EMBEDDING_TYPE == "openai":
        embed_model = OpenAIEmbedding(model=EMBEDDING_MODEL_NAME)
    else:
        raise NotImplementedError()        
    
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    logging.info("Load vectorstore successfully")
    # load the vectorstore
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context, embed_model=embed_model)
    query_engine = index.as_query_engine(llm =llm, streaming=stream, response_mode=RESPONSE_MODE)
    prompt_template, refine_template = get_prompt_template()
    query_engine.update_prompts(
        {"response_synthesizer:text_qa_template": prompt_template, "response_synthesizer:refine_template": refine_template}
    )

    return query_engine