import os 
import sys 
import ray
import logging
import json 
from typing import List
from ray import serve
import asyncio
from fastapi import FastAPI

# from langchain.llms import Ollama
from typing import List
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext

from starlette.responses import StreamingResponse, Response
import chromadb

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

# from googletrans import Translator

app = FastAPI()

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
        # self.translator = Translator()
        self.use_translate = cfg.MODEL.USE_TRANSLATE

    def setup_retrieval_qa_pipeline(self):
        # langchain.llm_cache = SQLiteCache(database_path=cfg.STORAGE.CACHE_DB_PATH)
        return self.create_retrieval_qa_pipeline(cfg.MODEL.DEVICE, cfg.MODEL.USE_HISTORY, cfg.MODEL.USE_RETRIEVER)

    async def agenerate_response(self, query):
        streaming_response = self.qa_pipeline.query(query)
        streaming_response.print_response_stream()
        for text in streaming_response.response_gen:
            print(text, end="", flush=True)
            yield text


    @app.post("/api/stream")
    async def get_streaming_response(self, query):
        print("Query: ", query)
        return StreamingResponse(self.agenerate_response(query), media_type="text/plain")

    @app.post("/api/generate")
    def generate_response(self, query) -> List[str]:   
        print("Query: ", query)
        return Response(str(self.qa_pipeline.query(query)))

        

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

        if use_ollama:
            llm = Ollama(model=model_id, temperature=cfg.MODEL.TEMPERATURE)
        else:
            raise NotImplementedError("The implementation for other types of LLMs are not ready yet!")    
        return llm
        

    def create_retrieval_qa_pipeline(self, device_type="cuda", use_history=False, use_retriever=True):
        """
        Initializes and returns a retrieval-based Question Answering (QA) pipeline.

        """

        llm = self.load_model(device_type, model_id=MODEL_ID, model_basename=MODEL_BASENAME, use_ollama=USE_OLLAMA)

        chroma_client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
        chroma_collection = chroma_client.get_or_create_collection("quickstart")
        embed_model = OllamaEmbedding(model_name="e5-mistral")
        
        
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        logging.info("Load vectorstore successfully")
        # load the vectorstore
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context, embed_model=embed_model)

        return index.as_query_engine(llm =llm, streaming=False)

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