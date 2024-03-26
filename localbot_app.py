import os 
import sys 
import ray
import logging
import json 
from typing import List
from ray import serve
import asyncio
from fastapi import FastAPI
import time 
# from langchain.llms import Ollama
from typing import List

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.llm_serving.model import load_model

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
import pandas as pd
from pandasai import SmartDataframe
from pandasai.llm import OpenAI as OpenAIPandas

from starlette.responses import StreamingResponse, Response

from src.engine.qa_engine import create_retrieval_qa_pipeline
from src.constants import (
    MODELS_PATH,
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
@serve.ingress(app)
class LocalBot:
    def __init__(self):
        os.environ["OMP_NUM_THREADS"] = "{}".format(cfg.RAY_CONFIG.OMP_NUM_THREADS)
        self.stream = cfg.MODEL.STREAM

        self.qa_pipeline = self.setup_retrieval_qa_pipeline()
        self.loop = asyncio.get_running_loop()
        # self.translator = Translator()
        self.use_translate = cfg.MODEL.USE_TRANSLATE

    def setup_retrieval_qa_pipeline(self):
        # langchain.llm_cache = SQLiteCache(database_path=cfg.STORAGE.CACHE_DB_PATH)
        return create_retrieval_qa_pipeline(cfg.MODEL.DEVICE, stream=self.stream)

    async def agenerate_response(self, streaming_response):
        if self.stream:
            for text in streaming_response.response_gen:
                yield text
        else:
            yield str(streaming_response)


    @app.post("/api/stream")
    def get_streaming_response(self, query):
        print("Query: ", query)
        streaming_response = self.qa_pipeline.query(query)
        return StreamingResponse(self.agenerate_response(streaming_response), media_type="text/plain")

    @app.post("/api/generate")
    def generate_response(self, query) -> List[str]:   
        print("Query: ", query)
        response = []
        if self.stream:
            streaming_response = self.qa_pipeline.query(query)
            print(streaming_response)
            for chunk in streaming_response.response_gen:
                response.append(str(chunk))
            return Response("".join(response))
        else:
            return Response(str(self.qa_pipeline.query(query)))

    
    @app.post("/api/data-analytic/")
    def generate_data_analytic(self, file: UploadFile = File(...), query: str = Form(...)):
        # Validate file is a csv
        print("FILE OBJ: ", file)
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="The uploaded file is not a CSV file.")
        
        data = pd.read_csv(file.file)
        openai_key = os.getenv('OPENAI_API_KEY')

        llm = OpenAIPandas(mode="gpt-3.5-turbo-0125", api_token=openai_key)
        smart_df = SmartDataframe(data, config={"llm": llm, "enable_cache": False})
        result = smart_df.chat(query)
        return Response(result)


logging.basicConfig(format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO)
if not os.path.exists(MODELS_PATH):
    os.makedirs(MODELS_PATH, exist_ok=True)

ray.init(
    object_store_memory=100 * 1024 * 1024,
    # log_to_driver=False,
    _system_config={
        "automatic_object_spilling_enabled": True,
        "object_spilling_config": json.dumps(
            {"type": "filesystem", "params": {"directory_path": "/tmp/spill"}},
            separators=(",", ":")
        )
    },
)

app_bot = LocalBot.bind()