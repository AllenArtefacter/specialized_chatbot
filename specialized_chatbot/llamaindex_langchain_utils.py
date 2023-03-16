# This script should contain some utilities for two python library for encapsulete openai
# LlamaIndex and LangChain
import llama_index
import langchain
from langchain.agents import ZeroShotAgent,AgentExecutor
from llama_index import PromptHelper, LLMPredictor
from langchain import OpenAI
from llama_index.prompts.prompts import QuestionAnswerPrompt

MODEL = "text-davinci-003"

LLM = OpenAI(
    temperature=0,
    model_name=MODEL,
    max_tokens=1600
)

LLM_PREDICTOR = LLMPredictor(llm=LLM)
PROMPT_HELPER = PromptHelper(4096, 1024, 30)

DEFAULT_TEXT_QA_PROMPT_TMPL = (
    "Given the context information and not prior knowledge, answer the question, and if the answer is not relevant to the context, said 'sorry I don't know' \n"
    "You are an expert to beer and be glad to recommand beer \n"
    "---------------------\n"
    "Context: {context_str}"
    "\n---------------------\n"
    "Please use the same language as in the question"
    "Qustion: {query_str}"
    "Answer:"
)
DEFAULT_TEXT_QA_PROMPT = QuestionAnswerPrompt(DEFAULT_TEXT_QA_PROMPT_TMPL)


def get_langchain_prompt_template(tools):
    prefix = """Given the context information and not prior knowledge, answer the question, and if the answer is not contained within the text below, said "sorry I don't know"""
    suffix = """
    You are an expert to beer and be glad to recommand beer

    Context: {context_str}

    Please use the same language
    Please finish the conversation:
    {query_str}
    """
    prompt = ZeroShotAgent.create_prompt(
        tools,
        prefix=prefix,
        suffix=suffix,
        input_variables=["query_str", "context_str"]
    )

    prompt_template = QuestionAnswerPrompt.from_langchain_prompt(prompt)

    return prompt_template
