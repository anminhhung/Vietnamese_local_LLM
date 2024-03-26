import logging
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI
from llama_index.llms.vllm import Vllm

def load_model(cfg, model_id, device_type="cpu", LOGGING=logging, service="ollama"):
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

    if service == "ollama":
        logging.info(f"Loading Ollama Model: {model_id}")
        llm = Ollama(model=model_id, temperature=cfg.MODEL.TEMPERATURE)
    elif service == "openai":
        logging.info(f"Loading OpenAI Model: {model_id}")
        llm = OpenAI(model=model_id, temperature=cfg.MODEL.TEMPERATURE)
    elif service == "vllm":
        logging.info(f"Loading VLLM Model: {model_id}")
        llm = Vllm(model=model_id, temperature=cfg.MODEL.TEMPERATURE)
    else:
        raise NotImplementedError("The implementation for other types of LLMs are not ready yet!")    
    return llm