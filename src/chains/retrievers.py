from langchain.schema.retriever import BaseRetriever

from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from typing import List 
from langchain.docstore.document import Document

class DummyRetriever(BaseRetriever):
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        return []

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        return []
