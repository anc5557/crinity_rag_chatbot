import os
import time
import streamlit as st
import logging
from dotenv import load_dotenv
from input_gspread import input_faq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_community.llms import Ollama
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain.chains.combine_documents import create_stuff_documents_chain

# 로깅 설정
logging.basicConfig(level=logging.INFO)

# .env 파일 로드
load_dotenv()

# 환경 변수에서 LLM 타입 및 Ollama URL 설정
ENV = os.getenv("ENV", "dev")

LLM_TYPE = "Ollama"  # "HuggingFace" 또는 "Ollama"

# Ollama URL 설정
if ENV == "dev":
    OLLAMA_BASE_URL = "http://localhost:11434"
elif ENV == "prod":
    OLLAMA_BASE_URL = "http://ollama:11434"

faiss_index_path = "db"
embedding_model_name = "jhgan/ko-sroberta-multitask"
huggingface_llm_model_name = "beomi/gemma-ko-2b"  # beomi/gemma-ko-2b
ollama_llm_model_name = "llama3.1:8b-instruct-q4_K_M"  # llama-3-Korean-Bllossom-8B-gguf-Q4_K_M or EEVE-Korean-10.8B-Q5_K_M-GGUF or EEVE-Korean-Instruct-10.8B-v1.0-GGUF-Q4-K-M or llama3.1:8b-instruct-q4_K_M


@st.cache_resource
def load_embedding_model(model_name):
    """임베딩 모델 로드"""
    embedding_model = HuggingFaceEmbeddings(
        model_name=model_name, encode_kwargs={"normalize_embeddings": True}
    )
    return embedding_model


@st.cache_resource
def load_vectorstore(index_path, _embedding_model):
    """벡터스토어 로드"""
    vectorstore = FAISS.load_local(
        folder_path=index_path,
        embeddings=_embedding_model,
        allow_dangerous_deserialization=True,
    )
    return vectorstore


@st.cache_resource
def load_llm(llm_model_name, llm_type):
    """LLM 로드
    llm_type: HuggingFace 또는 Ollama
    """
    if llm_type == "HuggingFace":
        llm = HuggingFacePipeline.from_model_id(
            model_id=llm_model_name,
            device=0,
            task="text-generation",
            pipeline_kwargs={
                "max_length": 1000,
                "num_return_sequences": 1,
                "max_new_tokens": 500,
            },
        )
    elif llm_type == "Ollama":
        llm = Ollama(model=llm_model_name, base_url=OLLAMA_BASE_URL)
    return llm


@st.cache_resource
def create_rag_chain(embedding_model_name, faiss_index_path, llm_model_name, llm_type):
    """RAG 체인 생성"""

    logging.info("서버 시작합니다.")
    embedding_model = load_embedding_model(embedding_model_name)
    logging.info("임베딩 모델 로드 완료")
    vectorstore = load_vectorstore(faiss_index_path, embedding_model)
    logging.info("벡터스토어 로드 완료")
    llm = load_llm(llm_model_name, llm_type)
    logging.info("LLM 로드 완료")

    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.5, "k": 3},
        verbose=True,
    )
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
    
    재구성된 질문만 응답하세요. 추가적인 텍스트는 포함하지 마세요.
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )

    return create_history_aware_retriever(llm, retriever, prompt)


def create_question_answering_chain(llm):
    """질문 답변 체인 생성"""

    system_prompt = """당신은 크리니티 Q&A 챗봇입니다. 검색된 문서를 기반으로 사용자의 질문에 답변하세요. 문서에 없는 정보는 만들어내지 마세요. 한국어로 답변해주세요. 세 문장 이내로 답변해주세요.모른다면, 모른다고 말해주세요. 검색된 문서가 없는 경우 "검색된 문서가 없습니다."라고 답변해주세요.
    ## 검색된 문서 ##
    {context}
    ################
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}\n"),
        ]
    )

    return create_stuff_documents_chain(llm, prompt)


def clean_data(data):
    """데이터 정제"""
    cleaned_data = []
    for item in data:
        cleaned_page_content = item.page_content.replace("\n", " ").strip()
        cleaned_data.append({"page_content": cleaned_page_content})
    return cleaned_data


def reset_chat():
    st.session_state.messages = []


def main():
    st.title("💭크리니티 Q&A 챗봇")

    with st.expander("알림", icon="📢", expanded=True):
        """사내 테스트 중입니다. 문서는 cm9.3 사용자 메뉴얼입니다. \n\n 문서 개선 작업중으로 내용이 부족하거나 부정확할 수 있습니다. \n\n 답변이 만족스럽지 않다면, [여기](https://docs.google.com/spreadsheets/d/1iu9H_OZPtnvGGXM07axeCWz8sH5eSSOpi84nrzMX5B0/edit?pli=1&gid=0#gid=0)를 클릭하여 '상이함'에 체크해주세요."""

    if LLM_TYPE == "HuggingFace":
        llm_model_name = huggingface_llm_model_name
    else:
        llm_model_name = ollama_llm_model_name

    # RAG 체인 생성
    rag_chain = create_rag_chain(
        embedding_model_name, faiss_index_path, llm_model_name, LLM_TYPE
    )
    st.session_state.rag_chain = rag_chain

    # 메세지가 없다면 []으로 선언
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 대화 초기화 버튼
    if st.button("대화 초기화"):
        reset_chat()
        st.toast("초기화 되었습니다.", icon="❌")

    with st.chat_message("assistant"):
        st.markdown(
            "안녕하세요! 크리니티 Q&A 챗봇입니다. 질문을 입력해주세요. 🤖 \n\n "
        )
        st.markdown("")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question!"):
        MAX_MESSAGES_BEFORE_DELETION = 4  # 최대 메세지 수

        if len(st.session_state.messages) >= MAX_MESSAGES_BEFORE_DELETION:
            del st.session_state.messages[:2]

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            with st.spinner("답변중입니다..."):
                rag_chain = st.session_state.rag_chain
                result = rag_chain.invoke(
                    {"input": prompt, "chat_history": st.session_state.messages}
                )

                st.session_state.messages.append({"role": "user", "content": prompt})

                cleaned_datas = clean_data(result["context"])

                for i, cleaned_data in enumerate(cleaned_datas):
                    with st.expander(f"참고 문서 {i+1}"):
                        st.write(cleaned_data["page_content"])

                for chunk in result["answer"].split(" "):
                    full_response += chunk + " "
                    time.sleep(0.2)
                    message_placeholder.markdown(full_response + "▌")
                    message_placeholder.markdown(full_response)

            st.session_state.messages.append(
                {"role": "assistant", "content": full_response}
            )

            # 질문과 답변을 스프레드시트에 기록
            if ENV == "prod":
                input_faq(prompt, full_response.strip())


main()
