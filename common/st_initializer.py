import os
from dotenv import load_dotenv
from initializer import Initializer
from agents.translation_agent import TranslationAgent
from agents.summarization_agent import SummarizationAgent
from agents.manual_qa_agent import ManualQaAgent
from agents.general_task_agent import GeneralTaskAgent
from agents.email_assistant_agent import EmailAssistantAgent
from router import Router

load_dotenv()

ENV = os.getenv("ENV", "dev")
OLLAMA_BASE_URL = (
    os.getenv("OLLAMA_BASE_URL") if ENV == "dev" else "http://ollama:11434"
)


def initialize_router():
    initializer = Initializer(
        embedding_model_name="bge-m3",
        index_path="db",
        llm_model_name="llama3.1:8b-instruct-q5_K_M",
        ollama_base_url=OLLAMA_BASE_URL,
        reranker_model_name="Dongjin-kr/ko-reranker",
    )
    llm, llm_json, retriever = initializer.initialize()

    translation = TranslationAgent(llm, llm_json)
    summarization = SummarizationAgent(llm)
    qa = ManualQaAgent(llm=llm, retriever=retriever)
    email_assistant = EmailAssistantAgent(llm)
    general_task = GeneralTaskAgent(llm)

    router = Router(
        llm_json, translation, summarization, qa, email_assistant, general_task
    )
    return translation, summarization, qa, email_assistant, general_task, router
