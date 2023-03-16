from typing import Any, Dict, Optional, Sequence, Type, cast, Union
import logging
import json
from .llamaindex_langchain_utils import (
    DEFAULT_TEXT_QA_PROMPT,
    MODEL,
    LLM,
    LLM_PREDICTOR,
    PROMPT_HELPER,
    get_langchain_prompt_template
)
from llama_index.indices.base import DOCUMENTS_INPUT, BaseGPTIndex
import json
from langchain import OpenAI
from llama_index.data_structs.data_structs import (
    ChromaIndexDict,
    FaissIndexDict,
    IndexDict,SimpleIndexDict,
)
from llama_index.embeddings.base import BaseEmbedding
from llama_index.indices.vector_store.base import GPTVectorStoreIndex
from llama_index.indices.base import DOCUMENTS_INPUT, BaseGPTIndex
from llama_index.indices.query.base import BaseGPTIndexQuery
from llama_index.indices.query.schema import QueryMode
from llama_index.vector_stores import (
    ChromaVectorStore,
    FaissVectorStore,
    PineconeVectorStore,
    QdrantVectorStore,
    SimpleVectorStore,
    WeaviateVectorStore,
)
from llama_index import (
    GPTSimpleVectorIndex,
    SimpleDirectoryReader,
    WikipediaReader,
    GPTListIndex,
    GPTKeywordTableIndex,
    GPTSimpleKeywordTableIndex,
    LLMPredictor, GPTSimpleVectorIndex, PromptHelper,
    QuestionAnswerPrompt
)
from llama_index.indices.query.vector_store.queries import (
    GPTChromaIndexQuery,
    GPTFaissIndexQuery,
    GPTSimpleVectorIndexQuery,

)
from langchain.agents import initialize_agent, Tool

# TODO:
# 1. A bot class. load data or load index, recieve query and ouput responese
# 2. A bot class. beyond 1, can also remember the conversation

import os
os.environ["TOKENIZERS_PARALLELISM"]= 'true'

class Chatbot(GPTVectorStoreIndex):
    """Chatbot based on text in directory
    Chatbot is inhrented from llama_index.GPTVectorStoreIndex
    And only accept data from a directory, where txt, pdf, doc,..etc may store in it
    For more info for the data type accepted, pls refer to https://gpt-index.readthedocs.io/en/latest/reference/readers.html#gpt_index.readers.SimpleDirectoryReader

    Paremeters:
    ----------
        document_directory: str
            data directory for you corpus hub, based on which the chatbot will answer

    Examples:
    ----------
    load from disk
    >>> bot = chatbot.Chatbot.load_from_disk('bot.json')
    conversations
    >>> bot.conversation("Budweiser?")
    continue conversation
    >>> bot.continue_conversation('Plese list the history for that company'))

    """

    index_struct_cls: Type[IndexDict] = SimpleIndexDict

    def __init__(
        self,
        document_directory: Optional[str]=None,
        documents: Optional[Sequence[DOCUMENTS_INPUT]] = None,
        index_struct: Optional[IndexDict] = None,
        text_qa_template: Optional[QuestionAnswerPrompt] = None,
        llm_predictor: Optional[LLMPredictor] = None,
        embed_model: Optional[BaseEmbedding] = None,
        simple_vector_store_data_dict: Optional[dict] = None,
        **kwargs: Any,
    ) -> None:
        """Init params."""
        vector_store = SimpleVectorStore(
            simple_vector_store_data_dict=simple_vector_store_data_dict
        )

        self.document_directory = document_directory
        if self.document_directory:
            documents = SimpleDirectoryReader(self.document_directory).load_data()

        #text_qa_template = DEFAULT_TEXT_QA_PROMPT
        llm_predictor = LLM_PREDICTOR
        prompt_helper = PROMPT_HELPER

        super().__init__(
            documents=documents,
            index_struct=index_struct,
            text_qa_template=text_qa_template,
            llm_predictor=llm_predictor,
            embed_model=embed_model,
            vector_store=vector_store,
            #prompt_helper = prompt_helper,
            **kwargs,
        )

        # TODO: Temporary hack to also store embeddings in index_struct
        embedding_dict = vector_store._data.embedding_dict
        self._index_struct.embeddings_dict = embedding_dict
        # update docstore with current struct
        self._docstore.add_documents([self.index_struct], allow_update=True)

        self.question_list = []
        self.answer_list = []

        self.tools = [
            Tool(
                name = "ABI Index",
                func=lambda q: str(self.query(q)),
                description="Answer with ABI content",
                return_direct=True
            ),
        ]

        self.langchain_prompt_template = get_langchain_prompt_template(self.tools)
        self.n_conversation = 5

    @classmethod
    def get_query_map(self) -> Dict[str, Type[BaseGPTIndexQuery]]:
        """Get query map."""
        return {
            QueryMode.DEFAULT: GPTSimpleVectorIndexQuery,
            QueryMode.EMBEDDING: GPTSimpleVectorIndexQuery,
        }

    def _preprocess_query(self, mode: QueryMode, query_kwargs: Any) -> None:
        """Preprocess query."""
        super()._preprocess_query(mode, query_kwargs)
        del query_kwargs["vector_store"]
        vector_store = cast(SimpleVectorStore, self._vector_store)
        query_kwargs["simple_vector_store_data_dict"] = vector_store._data

    def conversation(self, query, **kwargs):
        text = self.query(query,**kwargs)
        logging.info(query)
        return str(text)

    def continue_conversation(self, query, **kwargs):
        """answer with former conversation"""
        self.question_list.append(query)
        conversatiosn = self._concate_qa(query)
        #self.text_qa_template = self.langchain_prompt_template
        resonse = self.query(conversatiosn, **kwargs)
        self.answer_list.append(str(resonse))
        return str(resonse)

    def _concate_qa(self,query):
        len_counter = 0
        conversations = []
        if len(self.question_list):
            for q,a, in zip(
                self.question_list[-self.n_conversation:],
                self.answer_list[-self.n_conversation:]
            ):
                conversations.append(f"prompt:{q} \nresponse:{a} \n")
            text_conversations = '\n'.join(conversations)
            while len(text_conversations) >= 500 and len(conversations) >1:
                conversations = conversations[1:]
                text_conversations = '\n'.join(conversations)

            text_conversations += f"prompt:{query} \n response:"

            return text_conversations
