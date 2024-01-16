import os
import logging
# import click
import torch
import src.utils as utils
from langdetect import detect

import langchain 
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler  # for streaming response
from langchain.callbacks.manager import CallbackManager
from src.nlp_preprocessing import Translation
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers import ContextualCompressionRetriever
from langchain.schema.retriever import BaseRetriever, Document
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.cache import SQLiteCache
from langchain.llms.vllm import VLLM
from run_localGPT import load_model, retrieval_qa_pipline

from typing import List 

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

from src.prompt_template_utils import get_prompt_template

# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from transformers import (
    GenerationConfig,
    pipeline,
)

from src.load_models import (
    load_quantized_model_awq,
    load_quantized_model_gguf_ggml,
    load_quantized_model_qptq,
    load_full_model,
)

from src.constants import (
    EMBEDDING_MODEL_NAME,
    PERSIST_DIRECTORY,
    MODEL_ID,
    MODEL_BASENAME,
    MAX_NEW_TOKENS,
    MODELS_PATH,
    CHROMA_SETTINGS,
    cfg
)

import streamlit as st
from accelerate import Accelerator

@st.cache_resource()
def load_model(device_type="cuda", model_id="", model_basename=None, LOGGING=logging):
    return load_model(device_type, model_id=model_id, model_basename=model_basename, LOGGING=LOGGING)


@st.cache_resource()
def retrieval_qa_pipline(device_type="cpu", use_history=False, promptTemplate_type="llama", use_retriever=True):
    return retrieval_qa_pipline(device_type, use_history=use_history, promptTemplate_type=promptTemplate_type, use_retriever=use_retriever) 

def run_app(qa_pipeline):    
    st.title("💬 Chatbot")
    st.caption("🚀 I'm a Local Bot")
    
    # show source use to debug
    # show_sources = st.checkbox("Show sources", value=False)

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Tôi có thể giúp gì được cho bạn?"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])
        
    # Initialize the QA system using caching
    # translater = Translation(from_lang="en", to_lang='vi', mode='translate') 

    if query := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": query})
        st.chat_message("user").write(query)
        
        # Add spinner
        with st.spinner("Thinking..."):
            res = qa_pipeline(query)
            answer, docs = res["result"], res["source_documents"]

        # if translate_output:
        #     answer = translater(answer)
        
        st.session_state.messages.append({"role": "assistant", "content": answer})
        # st.chat_message("assistant").write(answer)
        
        response = answer
        
        # if show_sources:
        #     response += "\n\n"
        #     response += "----------------------------------SOURCE DOCUMENTS---------------------------\n"
        #     for document in docs:
        #         response += "\n> " + document.metadata["source"] + ":\n" + document.page_content
        #     response += "----------------------------------SOURCE DOCUMENTS---------------------------\n"
        
        # save_qa
        utils.log_to_csv(query, answer)
            
        st.chat_message("assistant").write(response)

if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO)

    if not os.path.exists(MODELS_PATH):
        os.mkdir(MODELS_PATH)

    langchain.llm_cache = SQLiteCache(database_path=cfg.STORAGE.CACHE_DB_PATH)
    qa = retrieval_qa_pipline(cfg.MODEL.DEVICE, cfg.MODEL.USE_HISTORY, cfg.MODEL.MODEL_TYPE, cfg.MODEL.USE_RETRIEVER)
    run_app(qa)