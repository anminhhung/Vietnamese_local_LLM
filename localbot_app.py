import os 
import sys 
import ray
import logging
import json 
from typing import List, Awaitable
from ray import serve
import asyncio
from fastapi import FastAPI

import langchain 
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler  # for streaming response
from langchain.callbacks.manager import CallbackManager, AsyncCallbackManager
from langchain.callbacks.base import BaseCallbackHandler

from src.nlp_preprocessing import Translation
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers import ContextualCompressionRetriever
from langchain.cache import SQLiteCache
from queue import Empty
from src.prompt_template_utils import get_prompt_template
from langchain.vectorstores import Chroma
from transformers import GenerationConfig, pipeline
from langchain.llms.vllm import VLLM, VLLMOpenAI
# from langchain.llms import Ollama
from typing import Any, Dict, List, Optional

# from langchain.callbacks.streaming_stdout_final_only import FinalStreamingStdOutCallbackHandler

from starlette.responses import StreamingResponse, Response
# from langchain.callbacks.streaming_aiter import AsyncIteratorCallbackHandler
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.chains.retrievers import DummyRetriever
from src.chains.pipeline import BatchRetrievalQA
from src.chains.llm_chains import StreamingVLLM

from src.load_models import (
    load_quantized_model_awq,
    load_quantized_model_gguf_ggml,
    load_quantized_model_qptq,
    load_full_model,
)
from asyncio import QueueEmpty
from src.chains.streaming import FinalStreamingStdOutCallbackHandler, MyCustomHandler, AsyncIteratorCallbackHandler
from src.llm.ollama_debug import Ollama
from src.constants import (
    EMBEDDING_MODEL_NAME,
    PERSIST_DIRECTORY,
    MODEL_ID,
    MODEL_BASENAME,
    MAX_NEW_TOKENS,
    MODELS_PATH,
    CHROMA_SETTINGS,
    USE_OLLAMA,
    cfg
)

from googletrans import Translator

handler = AsyncIteratorCallbackHandler()

app = FastAPI()


callback_manager = AsyncCallbackManager([handler])


@serve.deployment(
    ray_actor_options={"num_cpus": cfg.RAY_CONFIG.NUM_CPUS, 
                       "num_gpus": cfg.RAY_CONFIG.NUM_GPUS
    },
    max_concurrent_queries=cfg.RAY_CONFIG.MAX_CONCURRENT_QUERIES,
    autoscaling_config={
        "target_num_ongoing_requests_per_replica": cfg.RAY_CONFIG.NUM_REQUESTS_PER_REPLICA,
        "min_replicas": cfg.RAY_CONFIG.MIN_REPLICAS,
        "initial_replicas": cfg.RAY_CONFIG.INIT_REPLICAS,
        "max_replicas": cfg.RAY_CONFIG.MAX_REPLICAS,
    },
)
# @serve.deployment(num_replicas=1, ray_actor_options={"num_cpus": cfg.RAY_CONFIG.NUM_CPUS, 
#                        "num_gpus": cfg.RAY_CONFIG.NUM_GPUS})
@serve.ingress(app)
class LocalBot:
    def __init__(self):
        os.environ["OMP_NUM_THREADS"] = "{}".format(cfg.RAY_CONFIG.OMP_NUM_THREADS)
        self.qa_pipeline = self.setup_retrieval_qa_pipeline()
        self.loop = asyncio.get_running_loop()
        self.translator = Translator()
        self.use_translate = cfg.MODEL.USE_TRANSLATE

    def setup_retrieval_qa_pipeline(self):
        # langchain.llm_cache = SQLiteCache(database_path=cfg.STORAGE.CACHE_DB_PATH)
        return self.create_retrieval_qa_pipeline(cfg.MODEL.DEVICE, cfg.MODEL.USE_HISTORY, cfg.MODEL.MODEL_TYPE, cfg.MODEL.USE_RETRIEVER)

    # @serve.batch(max_batch_size=cfg.RAY_CONFIG.MAX_BATCH_SIZE, 
    #              batch_wait_timeout_s=cfg.RAY_CONFIG.BATCH_TIMEOUT
    # )

    async def consumer(self, queue):
        while True:
            item = await queue.get()
            if item is None:
                break
            yield item
            queue.task_done()

    async def agenerate_response(self, result):
        if USE_OLLAMA:
            for item in result:
                if self.use_translate:
                    item["result"] = self.translator.translate(item["result"], dest="vi").text

                for token in item['result'].split(' '):
                    print("item['result']: ",item['result'])

                    await asyncio.sleep(0.01)
                    yield token + ' '
        else:
            async def wrap_done(fn: Awaitable, event: asyncio.Event):
                """Wrap an awaitable with a event to signal when it's done or an exception is raised."""
                try:
                    await fn
                except Exception as e:
                #     # TODO: handle exception
                    print(f"Caught exception: {e}")
                finally:
                    # Signal the aiter to stop.
                    event.set()

            task = asyncio.create_task(wrap_done(self.qa_pipeline._acall(inputs=result, run_manager=callback_manager), handler.done))
            async for step in handler.aiter():
                yield [step]

            await task

    @app.post("/api/stream")
    async def get_streaming_response(self, query):
        print("Query: ", query)
        result = self.qa_pipeline.stream(query)

        return StreamingResponse(self.agenerate_response(result), media_type="text/plain")

    @app.post("/api/generate")
    def generate_response(self, query) -> List[str]:   
        print("Query: ", query)
        res = self.qa_pipeline(inputs=query)

        return Response(res["result"])

    def load_model(self, device_type="cpu", model_id="", model_basename=None, LOGGING=logging, use_ollama=False):
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

        if model_basename == "":
            model_basename = None
        if model_basename is not None:
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
                llm = VLLM(model=model_basename, trust_remote_code=True, max_new_tokens=MAX_NEW_TOKENS, temperature=cfg.MODEL.TEMPERATURE, top_k=10, top_p=0.95, quantization="awq", cache=False)
                return llm
            else:
                print("Load gptq model")
                llm = VLLM(model=model_basename, trust_remote_code=True, max_new_tokens=MAX_NEW_TOKENS, temperature=cfg.MODEL.TEMPERATURE, top_k=10, top_p=0.95, quantization="gptq", dtype='float16', cache=False)
                return llm
                #model, tokenizer = load_quantized_model_qptq(model_id, model_basename, device_type, LOGGING)
        else:
            print("load_full_model")
            logging.info(f"Using mode: {model_id}")
            
            if use_ollama:
                llm = Ollama(model=model_id, temperature=cfg.MODEL.TEMPERATURE, top_k=10, top_p=0.95, callbacks=[handler], cache=False)
            else:
                llm = StreamingVLLM(model=model_id, trust_remote_code=True, max_new_tokens=MAX_NEW_TOKENS, temperature=0.7, top_k=10, top_p=0.95, tensor_parallel_size=1, cache=False)
            return llm
            

    def create_retrieval_qa_pipeline(self, device_type="cuda", use_history=False, promptTemplate_type="llama", use_retriever=True):
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
        embeddings_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=cfg.MODEL.SIMILARITY_THRESHOLD)
        retriever = ContextualCompressionRetriever(
            base_compressor=embeddings_filter, base_retriever=retriever
        )

        if not use_retriever:
            retriever = DummyRetriever()

        # get the prompt template and memory if set by the user.
        prompt, memory = get_prompt_template(promptTemplate_type=promptTemplate_type, history=use_history)

        # load the llm pipeline
        llm = self.load_model(device_type, model_id=MODEL_ID, model_basename=MODEL_BASENAME, LOGGING=logging, use_ollama=USE_OLLAMA)
        
        qa_type = BatchRetrievalQA
        chain_type = "batch_stuff"
        if USE_OLLAMA:
            qa_type = RetrievalQA
            chain_type = "stuff"
            
        if use_history:
            qa = qa_type.from_chain_type(
                llm=llm,
                chain_type=chain_type,  # try other chains types as well. refine, map_reduce, map_rerank
                retriever=retriever,
                return_source_documents=False,  # verbose=True,
                callbacks=callback_manager,
                chain_type_kwargs={"prompt": prompt, "memory": memory},
            )
        else:
            qa = qa_type.from_chain_type(
                llm=llm,
                chain_type=chain_type,  # try other chains types as well. refine, map_reduce, map_rerank
                retriever=retriever,
                return_source_documents=False,  # verbose=True,
                callbacks=callback_manager,
                chain_type_kwargs={
                    "prompt": prompt,
                },
            )

        return qa

# if __name__ == "__main__":
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO)
if not os.path.exists(MODELS_PATH):
    os.makedirs(MODELS_PATH, exist_ok=True)

# ray.init(
#     _system_config={
#         "max_io_workers":4,
#         "min_spilling_size": 100*1024*1024, # Spill at least 100MB at a time
#         "object_spilling_config": json.dumps(
#             {
#                 "type": "filesystem",
#                 "params":{
#                     "directory_path": "/tmp/spill",
#                 },
#                 "buffer_size": 100*1024*1024, # use a 100MB buffer for writes
#             }
#         )
#     }
# )

ray.init(
    object_store_memory=100 * 1024 * 1024,
    _system_config={
        "automatic_object_spilling_enabled": True,
        "object_spilling_config": json.dumps(
            {"type": "filesystem", "params": {"directory_path": "/tmp/spill"}},
            separators=(",", ":")
        )
    },
)

app_bot = LocalBot.bind()