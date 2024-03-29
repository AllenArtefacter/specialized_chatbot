from typing import Any, Dict, Optional, Sequence, Type, cast, Union
import logging
import json
from .llamaindex_langchain_utils import (
    DEFAULT_TEXT_QA_PROMPT,
    HALF_OPENED_TEXT_QA_PROMPT_TMPL,
    MODEL,
    LLM,
    LLM_PREDICTOR,
    PROMPT_HELPER,
    get_langchain_prompt_template,
    get_llm_predictor
)
from .lang_utils import lang_detect
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

N_CONVERSATION_MEMORY = 3

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
        language_detect: bool
            set `language_detect` the bot will detect the language used by the question and add a
            prompt to tell OpenAI use the same language in the answer
        human_agent_name: str
            when comes a question what we feed to the chatbot is "{human_agent_name}:{quetions}\n{ai_agaent_name}:"
            e.g., if `human_agent_name` and `ai_angent_name` are set to be "Human" and "AI"
            the prompt will be:
            ```
            Please answer the question:
            Human: question
            AI:
            ```
        ai_angent_name: str
            when comes a question what we feed to the chatbot is "{human_agent_name}:{quetions}\n{ai_agaent_name}:"
            e.g., if `human_agent_name` and `ai_angent_name` are set to be "Human" and "AI"
            the prompt will be:
            ```
            Please answer the question:
            Human: question
            AI:
            ```
        n_conversation: int
            number of conversation the chatbot will track

    Examples:
    ----------
    load from disk
    >>> bot = chatbot.Chatbot.load_from_disk('bot.json')
    >>> bot.conversation("Budweiser?")
    >>> bot.continue_conversation('Plese list the history for that company'))

    """

    index_struct_cls: Type[IndexDict] = SimpleIndexDict

    def __init__(
        self,
        document_directory: Optional[str]=None,
        language_detect: Optional[bool]=False,
        human_agent_name: Optional[str] = 'prompt',
        ai_angent_name: Optional[str] = 'response',
        n_conversation: Optional[int] = N_CONVERSATION_MEMORY,
        documents: Optional[Sequence[DOCUMENTS_INPUT]] = None,
        index_struct: Optional[IndexDict] = None,
        prompt_helper: Optional[PromptHelper] = None,
        text_qa_template: Optional[QuestionAnswerPrompt] = None,
        llm_predictor: Optional[LLMPredictor] = None,
        embed_model: Optional[BaseEmbedding] = None,
        simple_vector_store_data_dict: Optional[dict] = None,
        **kwargs: Any,
    ) -> None:
        print('initializing chatbot')
        """Init params."""
        vector_store = SimpleVectorStore(
            simple_vector_store_data_dict=simple_vector_store_data_dict
        )

        self.document_directory = document_directory
        if self.document_directory:
            documents = SimpleDirectoryReader(self.document_directory).load_data()
        self.language_detect = language_detect
        #text_qa_template = DEFAULT_TEXT_QA_PROMPT
        if not llm_predictor:
            llm_predictor = LLM_PREDICTOR
        if not prompt_helper:
            prompt_helper = PROMPT_HELPER

        self.human_agent_name = human_agent_name
        self.ai_angent_name = ai_angent_name

        super().__init__(
            documents=documents,
            index_struct=index_struct,
            text_qa_template= DEFAULT_TEXT_QA_PROMPT,# text_qa_template,
            llm_predictor=llm_predictor,
            embed_model=embed_model,
            vector_store=vector_store,
            prompt_helper = prompt_helper,
            **kwargs,
        )

        # TODO: Temporary hack to also store embeddings in index_struct
        embedding_dict = vector_store._data.embedding_dict
        self._index_struct.embeddings_dict = embedding_dict
        # update docstore with current struct
        self._docstore.add_documents([self.index_struct], allow_update=True)
        # self.text_qa_template = self.langchain_prompt_template
        self.n_conversation = n_conversation

        self.question_list = []
        self.answer_list = []
        
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
        query = f"{self.human_agent_name}:{query}\n{self.ai_angent_name}:"
        if self.language_detect:
            lang_prompt = self._add_language_prompt(query)
            query = lang_prompt+query
        text = self.query(query,**kwargs)
        logging.info(query)
        # logging.info(text)
        return str(text)

    def continue_conversation(self, query, **kwargs):
        """answer with former conversation"""
        conversatiosn = self._concate_qa(query)
        if self.language_detect:
            lang_prompt = self._add_language_prompt(query)
            conversatiosn = lang_prompt + conversatiosn
        #self.text_qa_template = self.langchain_prompt_template
        resonse = self.query(conversatiosn, **kwargs)
        self.question_list.append(query)
        self.answer_list.append(str(resonse))
        return str(resonse)

    def _concate_qa(self,query)->str:
        text_conversations = ''
        conversations = []
        if len(self.answer_list):
            for q,a, in zip(
                self.question_list[-self.n_conversation:],
                self.answer_list[-self.n_conversation:]
            ):
                conversations.append(f"{self.human_agent_name}:{q} \n{self.ai_angent_name}:{a} \n")
            text_conversations = '\n'.join(conversations)
            while len(text_conversations) >= 1000 and len(conversations) >1:
                conversations = conversations[1:]
                text_conversations = '\n'.join(conversations)

        text_conversations += f"{self.human_agent_name}:{query} \n{self.ai_angent_name}:"

        return text_conversations

    def _add_language_prompt(self,query:str)->str:
        lang = lang_detect(query)
        if lang not in ['Chinese', 'English']:
            lang = 'English'
        logging.info(f"{lang} detected from {query}")
        lang_prompt = f"\nPlease do use {lang} to answer the final question\n"
        return lang_prompt
