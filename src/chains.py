# src/chains.py
import logging
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
    FewShotPromptTemplate,
)
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from models import load_embedding_model, load_vectorstore, load_llm
import config.config as config
from config.logging_config import *
import streamlit as st


@st.cache_resource
def create_rag_chain():
    logging.info("서버 시작합니다.")
    embedding_model = load_embedding_model(config.embedding_model_name)
    vectorstore = load_vectorstore(config.faiss_index_path, embedding_model)
    llm = load_llm(config.ollama_llm_model_name, config.LLM_TYPE)

    # Cross Encoder Reranker + 벡터스토어 retriever
    model = HuggingFaceCrossEncoder(model_name=config.reranker_model_name)
    compressor = CrossEncoderReranker(model=model, top_n=3)
    retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": 10}
        ),
    )

    logging.info("Hugging Face Cross Encoder 리랭커 적용 완료")
    question_rephrasing_chain = create_question_rephrasing_chain(llm, retriever)
    question_answering_chain = create_question_answering_chain(llm)
    rag_chain = create_retrieval_chain(
        question_rephrasing_chain, question_answering_chain
    )
    logging.info("RAG 체인 생성 완료")

    return rag_chain


def create_question_rephrasing_chain(llm, retriever):
    """질문 재구성 체인 생성"""
    system_prompt = """이전 대화 내용과 최신 사용자 질문이 있을 때, 이 질문이 이전 대화 내용과 관련이 있을 수 있습니다.
    이런 경우, 대화 내용을 알 필요 없이 독립적으로 이해할 수 있는 질문으로 바꾸세요.
    질문에 답할 필요는 없고, 필요하다면 그저 다시 구성하거나 그대로 두세요.
    
    재구성된 질문만 답변하세요.
    """

    # examples : few-shot learning examples
    examples = [
        {
            "chat_history": "Human: 메일 첨부파일 크기 제한이 있나요?\nAI: 일반 첨부파일의 경우 20MB, 대용량 파일 첨부의 경우 2GB까지 가능합니다.",
            "input": "형식에 제한이 있나요?",
            "rephrased": "메일 첨부파일 형식 제한이 있나요?",
        },
        {
            "chat_history": "Human: 일정 등록하는 방법을 알려줘\nAI: 일정 등록 버튼을 누르거나 날짜를 선택해 등록할 수 있습니다. 제목과 일시를 정한 후, 캘린더의 종류를 선택하고, 알람을 통해 미리 일정을 알릴 수 있습니다.",
            "input": "수정하는 방법은?",
            "rephrased": "일정 수정하는 방법은?",
        },
        {
            "chat_history": "Human: 주소록에 주소를 이동/복사하려면 어떻게 하나요?\nAI: 주소록에 주소를 이동/복사하려면, 주소록에서 주소를 선택한 후, 이동 또는 복사 버튼을 누르세요. 이동할 주소록을 선택하거나, 새로운 주소록을 만들어 이동할 수 있습니다.",
            "input": "복사하면 붙여넣기는 어떻게 하나요?",
            "rephrased": "주소록에 복사한 주소를 붙여넣는 방법은 무엇인가요?",
        },
        {
            "chat_history": "Human: 안녕\nAI: 안녕하세요! 무엇을 도와드릴까요?",
            "input": "메일 첨부파일 크기 제한이 있나요?",
            "rephrased": "메일 첨부파일 크기 제한이 있나요?",
        },
        {
            "chat_history": "Human: 일정 등록하는 방법을 알려줘 \nAI: 일정 등록 버튼을 누르거나 날짜를 선택해 등록할 수 있습니다. 제목과 일시를 정한 후, 캘린더의 종류를 선택하고, 알람을 통해 미리 일정을 알릴 수 있습니다.",
            "input": "메일 백업하는 방법을 알려줘",
            "rephrased": "메일 백업하는 방법을 알려줘",
        },
        {
            "chat_history": "Human: 메일 검색 기능에 대해 알고 싶어요.\nAI: 검색창에 검색어를 입력하면 전체, 보낸사람, 받는사람, 참조, 제목, 내용, 첨부파일명으로 메일을 검색할 수 있습니다.",
            "input": "제목으로 검색하는 방법은?",
            "rephrased": "메일 제목을 기준으로 검색하는 방법을 알려주세요.",
        },
    ]

    # MessagesPlaceholder 사용
    MessagesPlaceholder(variable_name="chat_history")

    # example_prompt
    example_prompt = PromptTemplate.from_template(
        "이전 대화 내용:\n{chat_history}\n최신 사용자 질문:\n{input}\n재구성된 질문:\n{rephrased}"
    )

    # few-shot prompt
    few_shot_prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix=system_prompt,
        suffix="이전 대화 내용:\n{chat_history}\n최신 사용자 질문:\n{input}\n재구성된 질문:\n",
        input_variables=["chat_history", "input"],
    )

    return create_history_aware_retriever(llm, retriever, few_shot_prompt)


def create_question_answering_chain(llm):
    """질문 답변 체인 생성"""
    system_prompt = """당신은 크리니티 Q&A 챗봇입니다. 검색된 문서를 기반으로 사용자의 질문에 답변하세요. 문서에 없는 정보는 만들어내지 마세요. 한국어로 답변해주세요. 모른다면, 모른다고 말해주세요.
    
    ## 검색된 문서 ##
    {context}
    #################
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}\n"),
        ]
    )

    return create_stuff_documents_chain(llm, prompt)
