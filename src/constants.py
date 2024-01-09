import os

# from dotenv import load_dotenv
from chromadb.config import Settings

# https://python.langchain.com/en/latest/modules/indexes/document_loaders/examples/excel.html?highlight=xlsx#microsoft-excel
from langchain.document_loaders import CSVLoader, PDFMinerLoader, TextLoader, UnstructuredExcelLoader, Docx2txtLoader
from langchain.document_loaders import UnstructuredFileLoader, UnstructuredMarkdownLoader, JSONLoader
from configs.config import get_config

cfg = get_config()
cfg.merge_from_file('configs/config_files/model.yaml')
cfg.merge_from_file('configs/config_files/storage.yaml')

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


# https://python.langchain.com/en/latest/_modules/langchain/document_loaders/excel.html#UnstructuredExcelLoader
DOCUMENT_MAP = {
    ".txt": TextLoader,
    ".md": UnstructuredMarkdownLoader,
    ".py": TextLoader,
    # ".pdf": PDFMinerLoader,
    ".pdf": UnstructuredFileLoader,
    ".csv": CSVLoader,
    ".xls": UnstructuredExcelLoader,
    ".xlsx": UnstructuredExcelLoader,
    ".docx": Docx2txtLoader,
    ".doc": Docx2txtLoader,
    ".json": JSONLoader,
}

EMBEDDING_MODEL_NAME = cfg.MODEL.EMBEDDING_MODEL_NAME
MODEL_ID  = cfg.MODEL.MODEL_ID
MODEL_BASENAME = cfg.MODEL.MODEL_BASENAME


