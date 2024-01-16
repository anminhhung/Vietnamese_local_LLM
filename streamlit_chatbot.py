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

def load_model(device_type, model_id, model_basename=None, LOGGING=logging):
    """
    Select a model for text generation using the HuggingFace library.
    If you are running this for the first time, it will download a model for you.
    subsequent runs will use the model from the disk.

    Args:
        device_type (str): Type of device to use, e.g., "cuda" for GPU or "cpu" for CPU.
        model_id (str): Identifier of the model to load from HuggingFace's model hub.
        model_basename (str, optional): Basename of the model if using quantized models.
            Defaults to None.

    Returns:
        HuggingFacePipeline: A pipeline object for text generation using the loaded model.

    Raises:
        ValueError: If an unsupported model or device type is provided.
    """
    logging.info(f"Loading Model: {model_id}, on: {device_type}")
    logging.info("This action can take a few minutes!")

    if model_basename is not None or model_basename != "":
        if ".gguf" in model_basename.lower():
            print("Load quantized model gguf")
            llm = load_quantized_model_gguf_ggml(model_id, model_basename, device_type, LOGGING)
            return llm
        elif ".ggml" in model_basename.lower():
            print("Load quantized model ggml")
            model, tokenizer = load_quantized_model_gguf_ggml(model_id, model_basename, device_type, LOGGING)
            # Load configuration from the model to avoid warnings
            # generation_config = GenerationConfig.from_pretrained(model_id)
            # see here for details:
            # https://huggingface.co/docs/transformers/
            # main_classes/text_generation#transformers.GenerationConfig.from_pretrained.returns

            # Create a pipeline for text generation
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_length=MAX_NEW_TOKENS,
                temperature=0.2,
                # top_p=0.95,
                repetition_penalty=1.15,
                # generation_config=generation_config,
            )

            local_llm = HuggingFacePipeline(pipeline=pipe)
            logging.info("Local LLM Loaded")

            return local_llm
        elif "awq" in model_basename.lower():
            #print("Load quantized model awq")
            #model, tokenizer = load_quantized_model_awq(model_id, LOGGING)
            llm = VLLM(model=model_basename, trust_remote_code=True, max_new_tokens=MAX_NEW_TOKENS, temperature=0.7, top_k=10, top_p=0.95, quantization="awq")
            return llm
        else:
            print("Load gptq model")
            llm = VLLM(model=model_basename, trust_remote_code=True, max_new_tokens=MAX_NEW_TOKENS, temperature=0.7, top_k=10, top_p=0.95, quantization="gptq", dtype='float16')
            return llm
            #model, tokenizer = load_quantized_model_qptq(model_id, model_basename, device_type, LOGGING)
    else:
        print("load_full_model")
        if device_type == "cpu":
            model, tokenizer = load_full_model(model_id, model_basename, device_type, LOGGING)
        else:
            llm = VLLM(model=model_id, trust_remote_code=True, max_new_tokens=MAX_NEW_TOKENS, temperature=0.7, top_k=10, top_p=0.95, tensor_parallel_size=2)
            return llm


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