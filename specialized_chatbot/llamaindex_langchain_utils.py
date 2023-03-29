# This script should contain some utilities for two python library for encapsulete openai
# LlamaIndex and LangChain
import llama_index
import langchain
from langchain.agents import ZeroShotAgent,AgentExecutor
from llama_index import PromptHelper, LLMPredictor
from langchain import OpenAI
from langchain.chat_models import ChatOpenAI
from llama_index.prompts.prompts import QuestionAnswerPrompt

MODEL = "text-davinci-003"
COMPETITORS = [
    "Heineken",
    "Molson Coors",
    "Carlsberg",
    "Constellation",
    "Pabst Brewing"
]




LLM = OpenAI(
    temperature=0,
    model_name=MODEL,
    max_tokens=256
)

LLM_PREDICTOR = LLMPredictor(llm=LLM)
PROMPT_HELPER = PromptHelper(2048, 256, 256)

DEFAULT_TEXT_QA_PROMPT_TMPL = (
    """Answer the question as truthfully as possible using the provided text, and if the answer is not contained within the text below, say "I don't know" \n"""
    #"Given the context information and not prior knowledge, answer the question, and if the answer is not relevant to the context, said 'sorry I don't know' \n"
    "You are an expert to beer and be glad to recommand beer \n"
    #f"You will never mention anything about {','.join(COMPETITORS)}\n"
    "Please don't mention any beer or brand if they are not in the givend context \n"
    "---------------------\n"
    "Context: {context_str}"
    "\n---------------------\n"
    "Based on the given context, finish the conversation:"
    "{query_str}"
)
DEFAULT_TEXT_QA_PROMPT = QuestionAnswerPrompt(DEFAULT_TEXT_QA_PROMPT_TMPL)


COMPETITORS = [
    "Heineken",
    "Molson Coors",
    "Carlsberg",
    "Constellation",
    "Pabst Brewing"
]

RECOMMANDERS = [
    "Boxing Cat",
    "Goose Island",
    "Budweiser (百威)",
    "Corona",
    "Kwak",
    "哈尔滨啤酒",
    "福佳",
    "百位黑金(Magnum)"
]

HALF_OPENED_TEXT_QA_PROMPT_TMPL = (
    #"Given the context information and not prior knowledge, answer the question, and if the answer is not relevant to the context, said 'sorry I don't know' \n"
    f"You will now play the role of the beer master of {','.join(RECOMMANDERS)} and always recommand beers of them. "
    "Based on this, you can chat with customers as a beer expert and be glad to recommand beers. "
    #f"If your answer will contains {','.join(COMPETITORS)} or not relevant to the given context, you will still recommand beers in the given context and expain why you recommand them\n"
    "If your answer is not relevant to the given context, you will try recommand beers in the given context and explain why you recommand them. "
    "You will always steer the conversation to the topics about given context. "
    "Please don't indicate you have a context in you answer.\n"
    """Answer the question using the provided text \n"""
    "---------------------\n"
    "Context: {context_str}"
    "\n---------------------\n"
    "Based on the given context, anwser the final question:\n"
    "{query_str}"
)
HALF_OPENED_TEXT_QA_PROMPT = QuestionAnswerPrompt(HALF_OPENED_TEXT_QA_PROMPT_TMPL)


HALF_OPENED_TEXT_QA_PROMPT_ZH_TMPL = (
    #"""Answer the question as truthfully as possible using the provided text \n"""
    #"Given the context information and not prior knowledge, answer the question, and if the answer is not relevant to the context, said 'sorry I don't know' \n"
    f"你现在将扮演{','.join(RECOMMANDERS)} 等品牌的啤酒大师与客人对话，并愉快地推荐这些品牌的啤酒。"
    f"如果你的回答可能与给定的情景无关，请还是推荐情境内提到的啤酒。"
    "你会将话题婉的拉回给定的情境中来。"
    "\n---------------------\n"
    "给定的情境： {context_str}"
    "\n---------------------\n"
    "基于给定的情境，请回答客人的问题:\n"
    "{query_str}"
)
HALF_OPENED_TEXT_QA_ZH_PROMPT = QuestionAnswerPrompt(HALF_OPENED_TEXT_QA_PROMPT_ZH_TMPL)

CHATBOT_PATH = 'half_opened.json'

RECOMMANDERS = [
    "Boxing Cat",
    "Goose Island",
    "Budweiser",
    "Corona",
    "Kwak",
    "Harbin beer",
    "Hoegaarden",
    "Magnum"
]

RECOMMANDERS_ZH = [
    "拳击猫",
    "鹅岛",
    "百威",
    "科罗娜",
    "比利时夸克啤酒",
    "哈尔滨啤酒",
    "福佳",
    "百威黑金"
]

MORE_OPENED_TEXT_QA_PROMPT_TMPL =(
    #"Given the context information and not prior knowledge, answer the question, and if the answer is not relevant to the context, said 'sorry I don't know' \n"
    f"You will now play the role of the beer master chat with human "
    "You will always find a way to recommand beers. "
    #f"If your answer will contains {','.join(COMPETITORS)} or not relevant to the given context, you will still recommand beers in the given context and expain why you recommand them\n"
    #"If your answer is not relevant to the given context, you will try recommand beers in the given context and explain why you recommand them. "
    #"You will always steer the conversation to the topics about given context. "
    f"If you are going to talk about beers, please talk more about beers in the given context or {', '.join(RECOMMANDERS)}\n"
    #"Please don't indicate you have a context in you answer.\n"
    #"""Answer the question using the provided text \n"""
    "---------------------\n"
    "Context: {context_str}"
    "\n---------------------\n"
    "Finish the conversation:\n"
    "{query_str}"
)
MORE_OPENED_TEXT_QA_PROMPT_ZH_TMPL =(
    #"Given the context information and not prior knowledge, answer the question, and if the answer is not relevant to the context, said 'sorry I don't know' \n"
    f"你将扮演啤酒大师与消费者聊天。 "
    "你总是可以找到合适的方法去推荐啤酒 "
    #f"If your answer will contains {','.join(COMPETITORS)} or not relevant to the given context, you will still recommand beers in the given context and expain why you recommand them\n"
    #"If your answer is not relevant to the given context, you will try recommand beers in the given context and explain why you recommand them. "
    #"You will always steer the conversation to the topics about given context. "
    f"如果你要谈论啤酒，尽量谈论情境中提到的啤酒，或者 {'，'.join(RECOMMANDERS_ZH)}\n"
    #"Please don't indicate you have a context in you answer.\n"
    #"""Answer the question using the provided text \n"""
    "---------------------\n"
    "情境: {context_str}"
    "\n---------------------\n"
    "请完成对话:\n"
    "{query_str}"
)


MORE_OPENED_TEXT_QA_PROMPT = QuestionAnswerPrompt(MORE_OPENED_TEXT_QA_PROMPT_TMPL)
MORE_OPENED_TEXT_QA_PROMPT_ZH = QuestionAnswerPrompt(MORE_OPENED_TEXT_QA_PROMPT_ZH_TMPL)


def get_llm_predictor(model_name:str=MODEL, **kwargs)->LLMPredictor:
    """get language model
    Parameters:
    -----------
    model_name, str
        model_name from openai,pls refer to https://platform.openai.com/docs/api-reference/completions
    kwargs:
        other arguments of OpenAI API
    """
    if model_name in ['gpt-3.5-turbo']:
        LLM = ChatOpenAI(model_name=model_name, **kwargs)
    else:
        LLM = OpenAI(model_name=model_name, **kwargs)

    llm_predictor = LLMPredictor(llm=LLM)
    return llm_predictor


def get_langchain_prompt_template(tools):
    prefix = """Given the context information and not prior knowledge, answer the question, make the answer short, and if the answer is not contained within the text below, said "sorry I don't know"""
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
