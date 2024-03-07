import os

# from dotenv import load_dotenv
from chromadb.config import Settings

# https://python.langchain.com/en/latest/modules/indexes/document_loaders/examples/excel.html?highlight=xlsx#microsoft-excel
# from langchain_community.document_loaders import CSVLoader, PDFMinerLoader, TextLoader, UnstructuredExcelLoader, Docx2txtLoader
# from langchain_community.document_loaders import UnstructuredFileLoader, UnstructuredMarkdownLoader, JSONLoader, MathpixPDFLoader
from configs.config import get_config

cfg = get_config()
cfg.merge_from_file('configs/config_files/model.yaml')
cfg.merge_from_file('configs/config_files/storage.yaml')
cfg.merge_from_file('configs/config_files/ray.yaml')
cfg.merge_from_file('configs/config_files/secret.yaml')

# load_dotenv()
ROOT_DIRECTORY = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")

# Define the folder for storing database
SOURCE_DIRECTORY = os.path.join(ROOT_DIRECTORY, cfg.STORAGE.SOURCE_DIRECTORY)

PERSIST_DIRECTORY = os.path.join(ROOT_DIRECTORY, cfg.STORAGE.DB)

MODELS_PATH = f"./{cfg.STORAGE.MODELS}"

# Can be changed to a specific number
INGEST_THREADS = os.cpu_count() or 8

# Define the Chroma settings
CHROMA_SETTINGS = Settings(
    anonymized_telemetry=False,
    is_persistent=True,
)

# Context Window and Max New Tokens
CONTEXT_WINDOW_SIZE = cfg.MODEL.CONTEXT_WINDOW_SIZE
MAX_NEW_TOKENS = int(CONTEXT_WINDOW_SIZE/4) #CONTEXT_WINDOW_SIZE  

#### If you get a "not enough space in the buffer" error, you should reduce the values below, start with half of the original values and keep halving the value until the error stops appearing

N_GPU_LAYERS = cfg.MODEL.N_GPU_LAYERS  # Llama-2-70B has 83 layers
N_BATCH = cfg.MODEL.N_BATCH

### From experimenting with the Llama-2-7B-Chat-GGML model on 8GB VRAM, these values work:
# N_GPU_LAYERS = 20
# N_BATCH = 512

EMBEDDING_MODEL_NAME = cfg.MODEL.EMBEDDING_MODEL_NAME
EMBEDDING_TYPE = cfg.MODEL.EMBEDDING_TYPE
MODEL_ID  = cfg.MODEL.MODEL_ID
MODEL_BASENAME = cfg.MODEL.MODEL_BASENAME
SERVICE = cfg.MODEL.SERVICE

SYSTEM_PROMPT = cfg.MODEL.SYSTEM_PROMPT

RESPONSE_MODE = cfg.MODEL.RESPONSE_MODE

USER_PROMPT = """
Ngữ cảnh được cung cấp như sau
---------------------
{context_str}
---------------------
Dựa trên nội dung được cung cấp. Hãy trả lời câu hỏi từ người dùng. Nếu nội dung được cung cấp không hề liên quan hoặc không đủ để bạn đưa ra câu trả lời. Hãy nói rằng bạn "Tôi không có đủ thông tin để trả lời".
Hãy trả lời và kết thúc câu trả lời một cách đầy đủ.

Nếu bạn ghi nhớ và làm đúng những gì tôi đã dặn dò, tôi sẽ tip cho bạn $1000 vào cuối ngày
User question: {query_str}
AI Response:

"""

REFINE_PROMPT = """
Câu hỏi gốc như sau: {query_str}
Chung ta có một câu trả lời có sẵn: {existing_answer}
Chúng tôi có cơ hội tinh chỉnh câu trả lời hiện có (chỉ khi cần) với một số ngữ cảnh khác bên dưới.
------------
{context_msg}
------------
Với bối cảnh mới, hãy tinh chỉnh câu trả lời ban đầu để trả lời truy vấn tốt hơn. Nếu ngữ cảnh không hữu ích, hãy trả lại câu trả lời ban đầu.
Refined Answer: 

"""

