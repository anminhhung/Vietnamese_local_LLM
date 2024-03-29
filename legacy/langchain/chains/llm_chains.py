from typing import Any, Dict, List, Optional, AsyncIterator

from langchain.callbacks.manager import CallbackManagerForLLMRun, AsyncCallbackManager, AsyncCallbackManagerForChainRun, AsyncCallbackManagerForLLMRun
from langchain.llms.base import BaseLLM
from langchain.llms.openai import BaseOpenAI
from langchain.pydantic_v1 import root_validator
from langchain.schema.output import Generation, LLMResult, GenerationChunk



class StreamingVLLM(BaseLLM):
    """VLLM language model."""

    model: str = ""
    """The name or path of a HuggingFace Transformers model."""

    tensor_parallel_size: Optional[int] = 1
    """The number of GPUs to use for distributed execution with tensor parallelism."""

    trust_remote_code: Optional[bool] = False
    """Trust remote code (e.g., from HuggingFace) when downloading the model 
    and tokenizer."""

    n: int = 1
    """Number of output sequences to return for the given prompt."""

    best_of: Optional[int] = None
    """Number of output sequences that are generated from the prompt."""

    presence_penalty: float = 0.0
    """Float that penalizes new tokens based on whether they appear in the 
    generated text so far"""

    frequency_penalty: float = 0.0
    """Float that penalizes new tokens based on their frequency in the 
    generated text so far"""

    temperature: float = 1.0
    """Float that controls the randomness of the sampling."""

    top_p: float = 1.0
    """Float that controls the cumulative probability of the top tokens to consider."""

    top_k: int = -1
    """Integer that controls the number of top tokens to consider."""

    use_beam_search: bool = False
    """Whether to use beam search instead of sampling."""

    stop: Optional[List[str]] = None
    """List of strings that stop the generation when they are generated."""

    ignore_eos: bool = False
    """Whether to ignore the EOS token and continue generating tokens after 
    the EOS token is generated."""

    max_new_tokens: int = 512
    """Maximum number of tokens to generate per output sequence."""

    logprobs: Optional[int] = None
    """Number of log probabilities to return per output token."""

    client: Any  #: :meta private:

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that python package exists in environment."""

        try:
            from src.llm.async_vllm import AsyncLLM as VLLModel
        except ImportError:
            raise ImportError(
                "Could not import vllm python package. "
                "Please install it with `pip install vllm`."
            )
        
        values["client"] = VLLModel(
            model=values["model"],
            tensor_parallel_size=values["tensor_parallel_size"],
            trust_remote_code=values["trust_remote_code"],
        )

        return values

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling vllm."""
        return {
            "n": self.n,
            "best_of": self.best_of,
            "max_tokens": self.max_new_tokens,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "temperature": self.temperature,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "stop": self.stop,
            "ignore_eos": self.ignore_eos,
            "use_beam_search": self.use_beam_search,
            "logprobs": self.logprobs,
        }

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Run the LLM on the given prompt and input."""

        from vllm import SamplingParams

        # build sampling parameters
        params = {**self._default_params, **kwargs, "stop": stop}
        sampling_params = SamplingParams(**params)
        # call the model
        outputs = self.client.generate(prompts, sampling_params)

        generations = []
        for output in outputs:
            text = output.outputs[0].text
            generations.append([Generation(text=text)])

        return LLMResult(generations=generations)
    
    async def _astream(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[GenerationChunk]:
        """Run the LLM on the given prompt and input."""

        from vllm import SamplingParams

        # build sampling parameters
        params = {**self._default_params, **kwargs, "stop": stop}
        sampling_params = SamplingParams(**params)
        # call the model
        

        results_generator = self.client.generate(prompts, sampling_params)

        # try:
        async for chunk in results_generator:
            yield GenerationChunk(text=chunk)
            if run_manager:
                await run_manager.on_llm_new_token(chunk)


        # except (KeyboardInterrupt, Exception) as e:
        #     await _run_manager.get_child().on_llm_error(e)
        #     raise e
        # else:
        #     await _run_manager.get_child().on_llm_end(LLMResult(generations=[[generation]]))

    async def agenerate_from_stream(self, 
        stream: AsyncIterator[str],
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
    ) -> LLMResult:
        """Async generate from a stream."""

        generation: Optional[GenerationChunk] = None
        async for chunk in stream:
            if generation is None:
                generation = chunk
            else:
                generation = chunk
        assert generation is not None
        
        result = LLMResult(generations=[[generation]])
        await run_manager.on_llm_end(result)

        return result

    async def _agenerate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Run the LLM on the given prompt and input."""
        print(run_manager)
        stream_iter = self._astream(prompts, stop, run_manager, **kwargs)
        return await self.agenerate_from_stream(stream_iter, run_manager=run_manager)

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "vllm"