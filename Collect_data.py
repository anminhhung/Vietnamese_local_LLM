import os
import logging
# import click
import torch
import src.utils as utils
from langdetect import detect

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

class DummyRetriever(BaseRetriever):
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        return []

@st.cache_resource()
def load_model(device_type="cpu", model_id="", model_basename=None, LOGGING=logging):
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

    if model_basename is not None:
        if ".gguf" in model_basename.lower():
            print("Load quantized model gguf")
            llm = load_quantized_model_gguf_ggml(model_id, model_basename, device_type, LOGGING)
            return llm
        elif ".ggml" in model_basename.lower():
            print("Load quantized model ggml")
            model, tokenizer = load_quantized_model_gguf_ggml(model_id, model_basename, device_type, LOGGING)
        elif ".awq" in model_basename.lower():
            print("Load quantized model awq")
            model, tokenizer = load_quantized_model_awq(model_id, LOGGING)
        else:
            print("Load quantized model qptq")
            model, tokenizer = load_quantized_model_qptq(model_id, model_basename, device_type, LOGGING)
    else:
        print("load_full_model")
        model, tokenizer = load_full_model(model_id, model_basename, device_type, LOGGING)

    # Load configuration from the model to avoid warnings
    generation_config = GenerationConfig.from_pretrained(model_id)
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
        generation_config=generation_config,
    )

    local_llm = HuggingFacePipeline(pipeline=pipe)
    logging.info("Local LLM Loaded")

    return local_llm


@st.cache_resource()
def retrieval_qa_pipline(device_type="cpu", use_history=False, promptTemplate_type="llama", use_retriever=True):
    """
    Initializes and returns a retrieval-based Question Answering (QA) pipeline.

    This function sets up a QA system that retrieves relevant information using embeddings
    from the HuggingFace library. It then answers questions based on the retrieved information.

    Parameters:
    - device_type (str): Specifies the type of device where the model will run, e.g., 'cpu', 'cuda', etc.
    - use_history (bool): Flag to determine whether to use chat history or not.

    Returns:
    - RetrievalQA: An initialized retrieval-based QA system.

    Notes:
    - The function uses embeddings from the HuggingFace library, either instruction-based or regular.
    - The Chroma class is used to load a vector store containing pre-computed embeddings.
    - The retriever fetches relevant documents or data based on a query.
    - The prompt and memory, obtained from the `get_prompt_template` function, might be used in the QA system.
    - The model is loaded onto the specified device using its ID and basename.
    - The QA system retrieves relevant documents using the retriever and then answers questions based on those documents.
    """

    embeddings = HuggingFaceInstructEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={"device": device_type})
    # uncomment the following line if you used HuggingFaceEmbeddings in the ingest.py
    # embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    # load the vectorstore
    db = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings,
        client_settings=CHROMA_SETTINGS
    )
    retriever = db.as_retriever()
    embeddings_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.7)
    retriever = ContextualCompressionRetriever(
        base_compressor=embeddings_filter, base_retriever=retriever
    )

    if not use_retriever:
        retriever = DummyRetriever()

    # get the prompt template and memory if set by the user.
    prompt, memory = get_prompt_template(promptTemplate_type=promptTemplate_type, history=use_history)

    # load the llm pipeline
    llm = load_model(device_type, model_id=MODEL_ID, model_basename=MODEL_BASENAME, LOGGING=logging)

    if use_history:
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",  # try other chains types as well. refine, map_reduce, map_rerank
            retriever=retriever,
            return_source_documents=True,  # verbose=True,
            callbacks=callback_manager,
            chain_type_kwargs={"prompt": prompt, "memory": memory},
        )
    else:
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",  # try other chains types as well. refine, map_reduce, map_rerank
            retriever=retriever,
            return_source_documents=True,  # verbose=True,
            callbacks=callback_manager,
            chain_type_kwargs={
                "prompt": prompt,
            },
        )

    return qa

def pmAsk(qa_pipeline, content):
    baRole = "Bạn là một Business Analysist của công ty, bạn có nhiệm vụ phân tích chi tiết yêu cầu từ PM (Project Manager) và người dùng (User). Sau khi có đủ yêu cầu và dữ liệu sẽ tiến hành phân tích, ra danh sách chức năng, rồi phân tích từng chức năng ra chi tiết để Developer có thể tiến hành code. "
    #PM Ask BA to work
    pm_request = content
    st.session_state.messages.append({"role": "PM", "content": pm_request})
    # Add spinner
    with st.spinner("Thinking..."):
        res = qa_pipeline(baRole + pm_request)
        answer, docs = res["result"], res["source_documents"]
    if answer == None:
        pm_reply = "Bạn còn điều gì thông tin gì không?"
        st.chat_message("BA",avatar='👩🏻‍🦰').write(pm_reply)
        baAsk(qa_pipeline,pm_reply)
    else:
        if answer == "":
            pm_reply = "Bạn còn điều gì thông tin gì không?"
            st.chat_message("BA",avatar='👩🏻‍🦰').write(pm_reply)
            baAsk(qa_pipeline,pm_reply)
        else:
            st.session_state.messages.append({"role": "BA", "content": answer})
            response = answer
            # save_qa
            utils.log_to_csv(pm_request, answer)
            st.chat_message("BA",avatar='👩🏻‍🦰').write(response)
            pm_reply = "Tôi vừa nêu ý kiến, bạn cần cải thiện gì không?"
            st.chat_message("BA",avatar='👩🏻‍🦰').write(pm_reply)
            baAsk(qa_pipeline,pm_reply)

def baAsk(qa_pipeline, content):
    pmRole = "Bạn là một Project Manager của công ty, bạn phải đưa ra quyết định cho phân tích từ Business Analysist để xác nhận xem đúng theo ý muốn của khách hàng hay không và đưa ra từng bước tiến hành để thực hiện yêu cầu. Bạn phải yêu cầu Business Analysist đưa ra danh sách chức năng của website theo yêu cầu người dùng, yêu cầu Business Analysist phân tích chi tiết từng chức năng trong danh sách. "
    #BA Ask PM to work
    ba_request = content
    st.session_state.messages.append({"role": "BA", "content": ba_request})
    # Add spinner
    with st.spinner("Thinking..."):
        res = qa_pipeline(pmRole + ba_request)
        answer, docs = res["result"], res["source_documents"]
    if answer == None:
        ba_reply = "Bạn tiến hành liệt kê để tính hành thực hiện yêu cầu ban đầu đi!"
        st.chat_message("PM",avatar='🧔🏻').write(ba_reply)
        pmAsk(qa_pipeline,ba_reply)
    else:
        if answer == "":
            ba_reply = "Bạn tiến hành liệt kê để tính hành thực hiện yêu cầu ban đầu đi!"
            st.chat_message("PM",avatar='🧔🏻').write(ba_reply)
            pmAsk(qa_pipeline,ba_reply)
        else:
            st.session_state.messages.append({"role": "PM", "content": answer})
            response = answer
            # save_qa
            utils.log_to_csv(ba_request, answer)
            st.chat_message("PM",avatar='🧔🏻').write(response)
            ba_reply = "Tôi vừa nêu ý kiến, bạn cần cải thiện gì không?"
            st.chat_message("PM",avatar='🧔🏻').write(ba_reply)
            pmAsk(qa_pipeline,ba_reply)

def run_app(qa_pipeline):    
    st.title("🏢 Công ty TNHH Chạy Bằng Cơm")
    st.caption("🌏 Phòng phát triển")
    
    # show source use to debug
    # show_sources = st.checkbox("Show sources", value=False)

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "PM", "content": "Lãnh đạo hỏi, chúng tôi trả lời!"}]

    for msg in st.session_state.messages:
        if msg["role"] == "PM":
            st.chat_message(msg["role"],avatar='🧔🏻').write(msg["content"])
        else:
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
        
        st.session_state.messages.append({"role": "PM", "content": answer})
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
        st.chat_message("PM",avatar='🧔🏻').write(response)
        ba_request = "Hãy phân tích yêu cầu trên!"
        st.chat_message("PM",avatar='🧔🏻').write(ba_request)
        pmAsk(qa_pipeline, ba_request)


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO)

    if not os.path.exists(MODELS_PATH):
        os.mkdir(MODELS_PATH)

    qa = retrieval_qa_pipline(cfg.MODEL.DEVICE, cfg.MODEL.USE_HISTORY, cfg.MODEL.MODEL_TYPE, cfg.MODEL.USE_RETRIEVER)
    run_app(qa)