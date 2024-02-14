from langchain.chains.retrieval_qa.base import BaseRetrievalQA
from langchain.schema.retriever import BaseRetriever
from langchain.docstore.document import Document

import inspect
from langchain.pydantic_v1 import Extra, Field, root_validator
from typing import Any, Dict, List, Optional, Tuple
from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
    Callbacks,
)
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks.base import BaseCallbackManager

from langchain.chains.combine_documents.base import (
    BaseCombineDocumentsChain,
)
from langchain.chains.llm import LLMChain
from langchain.prompts.prompt import PromptTemplate
from langchain.schema import BasePromptTemplate, format_document
from langchain.schema.language_model import BaseLanguageModel
from langchain.chains.question_answering import (
    map_reduce_prompt,
    refine_prompts,
    stuff_prompt,
)

def _get_default_document_prompt() -> PromptTemplate:
    return PromptTemplate(input_variables=["page_content"], template="{page_content}")


class BatchRetrievalQA(BaseRetrievalQA):
    """Chain for question-answering against an index.

    Example:
        .. code-block:: python

            from langchain.llms import OpenAI
            from langchain.chains import RetrievalQA
            from langchain.faiss import FAISS
            from langchain.vectorstores.base import VectorStoreRetriever
            retriever = VectorStoreRetriever(vectorstore=FAISS(...))
            retrievalQA = RetrievalQA.from_llm(llm=OpenAI(), retriever=retriever)

    """

    retriever: BaseRetriever = Field(exclude=True)

    def _get_docs(
        self,
        question: str,
        *,
        run_manager: CallbackManagerForChainRun,
    ) -> List[List[Document]]:
        """Get docs."""
        return self.retriever.get_relevant_documents(
            question, callbacks=run_manager
        )

    async def _aget_docs(
        self,
        question: str,
        *,
        run_manager: AsyncCallbackManagerForChainRun,
    ) -> List[Document]:
        """Get docs."""
        return await self.retriever.aget_relevant_documents(
            question, callbacks=run_manager
        )


    @classmethod
    def from_chain_type(
        cls,
        llm: BaseLanguageModel,
        chain_type: str = "stuff",
        chain_type_kwargs: Optional[dict] = None,
        **kwargs: Any,
    ) -> BaseRetrievalQA:
        """Load chain from chain type."""
        _chain_type_kwargs = chain_type_kwargs or {}
        if chain_type == "batch_stuff":
            combine_documents_chain = _load_batch_stuff_chain(
                llm, **_chain_type_kwargs
            )
        else:
            combine_documents_chain = load_qa_chain(
                llm, chain_type=chain_type, **_chain_type_kwargs
            )
        return cls(combine_documents_chain=combine_documents_chain, **kwargs)

    async def _acall(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """Run get_relevant_text and llm on input query.

        If chain has 'return_source_documents' as 'True', returns
        the retrieved documents as well under the key 'source_documents'.

        Example:
        .. code-block:: python

        res = indexqa({'query': 'This is my query'})
        answer, docs = res['result'], res['source_documents']
        """
        questions = inputs
        if not isinstance(inputs, List):
            questions = [inputs]

        accepts_run_manager = (
            "run_manager" in inspect.signature(self._aget_docs).parameters
        )
        if accepts_run_manager:
            docs = [await self._aget_docs(question, run_manager=run_manager) for question in questions]
        else:
            docs = [await self._aget_docs(question) for question in questions]  # type: ignore[call-arg]
        answer = await self.combine_documents_chain.arun(
            input_documents=docs, question=questions, callbacks=run_manager
        )

        if self.return_source_documents:
            return {self.output_key: answer, "source_documents": docs}
        else:
            return {self.output_key: answer}


    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """Run get_relevant_text and llm on input query.

        If chain has 'return_source_documents' as 'True', returns
        the retrieved documents as well under the key 'source_documents'.

        Example:
        .. code-block:: python

        res = indexqa({'query': 'This is my query'})
        answer, docs = res['result'], res['source_documents']
        """
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        questions = inputs[self.input_key]
        accepts_run_manager = (
            "run_manager" in inspect.signature(self._get_docs).parameters
        )
        if accepts_run_manager:
            docs = [self._get_docs(question, run_manager=_run_manager) for question in questions]
        else:
            docs = [self._get_docs(question, run_manager=_run_manager) for question in questions]
        
        answer = self.combine_documents_chain.run(
            input_documents=docs, question=questions, callbacks=_run_manager
        )

        if self.return_source_documents:
            return {self.output_key: answer, "source_documents": docs}
        else:
            return {self.output_key: answer}

    @property
    def _chain_type(self) -> str:
        """Return the chain type."""
        return "retrieval_qa"



class BatchStuffDocumentsChain(BaseCombineDocumentsChain):
    """Chain that combines documents by stuffing into context.

    This chain takes a list of documents and first combines them into a single string.
    It does this by formatting each document into a string with the `document_prompt`
    and then joining them together with `document_separator`. It then adds that new
    string to the inputs with the variable name set by `document_variable_name`.
    Those inputs are then passed to the `llm_chain`.

    Example:
        .. code-block:: python

            from langchain.chains import StuffDocumentsChain, LLMChain
            from langchain.prompts import PromptTemplate
            from langchain.llms import OpenAI

            # This controls how each document will be formatted. Specifically,
            # it will be passed to `format_document` - see that function for more
            # details.
            document_prompt = PromptTemplate(
                input_variables=["page_content"],
                 template="{page_content}"
            )
            document_variable_name = "context"
            llm = OpenAI()
            # The prompt here should take as an input variable the
            # `document_variable_name`
            prompt = PromptTemplate.from_template(
                "Summarize this content: {context}"
            )
            llm_chain = LLMChain(llm=llm, prompt=prompt)
            chain = StuffDocumentsChain(
                llm_chain=llm_chain,
                document_prompt=document_prompt,
                document_variable_name=document_variable_name
            )
    """

    llm_chain: LLMChain
    """LLM chain which is called with the formatted document string,
    along with any other inputs."""
    document_prompt: BasePromptTemplate = Field(
        default_factory=_get_default_document_prompt
    )
    """Prompt to use to format each document, gets passed to `format_document`."""
    document_variable_name: str
    """The variable name in the llm_chain to put the documents in.
    If only one variable in the llm_chain, this need not be provided."""
    document_separator: str = "\n\n"
    """The string with which to join the formatted documents"""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @root_validator(pre=True)
    def get_default_document_variable_name(cls, values: Dict) -> Dict:
        """Get default document variable name, if not provided.

        If only one variable is present in the llm_chain.prompt,
        we can infer that the formatted documents should be passed in
        with this variable name.
        """
        llm_chain_variables = values["llm_chain"].prompt.input_variables
        if "document_variable_name" not in values:
            if len(llm_chain_variables) == 1:
                values["document_variable_name"] = llm_chain_variables[0]
            else:
                raise ValueError(
                    "document_variable_name must be provided if there are "
                    "multiple llm_chain_variables"
                )
        else:
            if values["document_variable_name"] not in llm_chain_variables:
                raise ValueError(
                    f"document_variable_name {values['document_variable_name']} was "
                    f"not found in llm_chain input_variables: {llm_chain_variables}"
                )
        return values

    @property
    def input_keys(self) -> List[str]:
        extra_keys = [
            k for k in self.llm_chain.input_keys if k != self.document_variable_name
        ]
        return super().input_keys + extra_keys

    def _get_inputs(self, docs: List[List[Document]], **kwargs: Any) -> dict:
        """Construct inputs from kwargs and docs.

        Format and the join all the documents together into one input with name
        `self.document_variable_name`. The pluck any additional variables
        from **kwargs.

        Args:
            docs: List of documents to format and then join into single input
            **kwargs: additional inputs to chain, will pluck any other required
                arguments from here.

        Returns:
            dictionary of inputs to LLMChain
        """
        # Format each document according to the prompt
        doc_strings = [[format_document(doc, self.document_prompt) for d in doc] for doc in docs]
        # Join the documents together to put them in the prompt.
        inputs = {
            k: v
            for k, v in kwargs.items()
            if k in self.llm_chain.prompt.input_variables
        }
        inputs[self.document_variable_name] = [self.document_separator.join(d_strings) for d_strings in doc_strings] 
        inputs = [{k: v[i] for k, v in inputs.items()} for i in range(len(inputs[self.document_variable_name]))]
        return inputs

    def prompt_length(self, docs: List[List[Document]], **kwargs: Any) -> Optional[int]:
        """Return the prompt length given the documents passed in.

        This can be used by a caller to determine whether passing in a list
        of documents would exceed a certain prompt length. This useful when
        trying to ensure that the size of a prompt remains below a certain
        context limit.

        Args:
            docs: List[Document], a list of documents to use to calculate the
                total prompt length.

        Returns:
            Returns None if the method does not depend on the prompt length,
            otherwise the length of the prompt in tokens.
        """
        inputs = self._get_inputs(docs, **kwargs)
        prompts = [self.llm_chain.prompt.format(**input) for input in inputs]
        return [self.llm_chain.llm.get_num_tokens(prompt) for prompt in prompts]


    def combine_docs(
        self, docs: List[List[Document]], callbacks: Callbacks = None, **kwargs: Any
    ) -> Tuple[str, dict]:
        """Stuff all documents into one prompt and pass to LLM.

        Args:
            docs: List of documents to join together into one variable
            callbacks: Optional callbacks to pass along
            **kwargs: additional parameters to use to get inputs to LLMChain.

        Returns:
            The first element returned is the single string output. The second
            element returned is a dictionary of other keys to return.
        """
        inputs = self._get_inputs(docs, **kwargs)
        # Call predict on the LLM.
        return self.llm_chain.create_outputs(self.llm_chain.generate(inputs)), {}

    async def acombine_docs(
        self, docs: List[Document], callbacks: Callbacks = None, **kwargs: Any
    ) -> Tuple[str, dict]:
        """Async stuff all documents into one prompt and pass to LLM.

        Args:
            docs: List of documents to join together into one variable
            callbacks: Optional callbacks to pass along
            **kwargs: additional parameters to use to get inputs to LLMChain.

        Returns:
            The first element returned is the single string output. The second
            element returned is a dictionary of other keys to return.
        """
        inputs = self._get_inputs(docs, **kwargs)
        # Call predict on the LLM.
        print("CALLBACKS A: ", callbacks)
        output = await self.llm_chain.agenerate(inputs, run_manager=callbacks)
        return output, {}

    async def _acall(
        self,
        inputs: Dict[str, List[Document]],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        """Prepare inputs, call combine docs, prepare outputs."""
        _run_manager = run_manager or AsyncCallbackManagerForChainRun.get_noop_manager()
        docs = inputs[self.input_key]
        # Other keys are assumed to be needed for LLM prediction
        other_keys = {k: v for k, v in inputs.items() if k != self.input_key}
        output, extra_return_dict = await self.acombine_docs(
            docs, callbacks=_run_manager, **other_keys
        )
        extra_return_dict[self.output_key] = output
        return extra_return_dict

    @property
    def _chain_type(self) -> str:
        return "stuff_documents_chain"

def _load_batch_stuff_chain(
    llm: BaseLanguageModel,
    prompt: Optional[BasePromptTemplate] = None,
    document_variable_name: str = "context",
    verbose: Optional[bool] = None,
    callback_manager: Optional[BaseCallbackManager] = None,
    callbacks: Callbacks = None,
    **kwargs: Any,
) -> BatchStuffDocumentsChain:
    _prompt = prompt or stuff_prompt.PROMPT_SELECTOR.get_prompt(llm)
    llm_chain = LLMChain(
        llm=llm,
        prompt=_prompt,
        verbose=verbose,
        callback_manager=callback_manager,
        callbacks=callbacks,
    )
    # TODO: document prompt
    return BatchStuffDocumentsChain(
        llm_chain=llm_chain,
        document_variable_name=document_variable_name,
        verbose=verbose,
        callback_manager=callback_manager,
        callbacks=callbacks,
        **kwargs,
    )
