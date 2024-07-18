import time
import streamlit as st
import logging
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_community.llms import Ollama
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain.chains.combine_documents import create_stuff_documents_chain

# 로깅 설정
logging.basicConfig(level=logging.INFO)


faiss_index_path = "db"
embedding_model_name = "jhgan/ko-sroberta-multitask"
llm_model_name = "EEVE-Korean-10.8B-Q5_K_M-GGUF"


@st.cache_resource
def load_embedding_model(model_name):
    embedding_model = HuggingFaceEmbeddings(model_name=model_name)
    return embedding_model


@st.cache_resource
def load_vectorstore(index_path, _embedding_model):
    vectorstore = FAISS.load_local(
        folder_path=index_path,
        embeddings=_embedding_model,
        allow_dangerous_deserialization=True,
    )
    return vectorstore


@st.cache_resource
def load_llm(llm_model_name):
    llm = Ollama(model=llm_model_name)
    return llm


@st.cache_resource
def create_rag_chain(embedding_model_name, faiss_index_path, llm_model_name):
    logging.info("서버 시작합니다.")

    embedding_model = load_embedding_model(embedding_model_name)
    logging.info("임베딩 모델 로드 완료")
    vectorstore = load_vectorstore(faiss_index_path, embedding_model)
    logging.info("벡터스토어 로드 완료")
    llm = load_llm(llm_model_name)
    logging.info("LLM 로드 완료")
    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 2}
    )
    question_rephrasing_chain = create_question_rephrasing_chain(llm, retriever)
    question_answering_chain = create_question_answering_chain(llm)
    rag_chain = create_retrieval_chain(
        question_rephrasing_chain, question_answering_chain
    )
    logging.info("RAG 체인 생성 완료")
    return rag_chain


def create_question_rephrasing_chain(llm, retriever):
    system_prompt = """
    당신은 질문 재구성자입니다. 이전 대화 내용과 최신 사용자 질문이 있을 때, 이 질문이 이전 대화 내용과 관련이 있을 수 있습니다.
    이런 경우, 대화 내용을 알 필요 없이 독립적으로 이해할 수 있는 질문으로 바꾸세요. 
    이 재구성된 질문은 문서 검색에만 사용됩니다. 사용자에게 제공할 최종 답변에는 영향을 미치지 않습니다.
    관련이 없는 경우, 질문을 그대로 두세요. 절대 질문에 답변을 제공하지 마세요.
    
    예시:
    관련 있는 경우)
    Human: 메일을 백업하고 싶어
    AI: 메일 백업은 기본메일함 관리 > 내 메일함 관리에서 가능합니다. 다운로드 버튼을 이용해 메일함을 zip 파일로 다운로드할 수 있습니다. 원하는 기간의 메일을 백업하려면, 기간별 백업을 체크하세요. 백업한 메일은 다운로드한 파일을 업로드하여 다시 가져올 수 있습니다.
    Human: 업로드는 어떻게 하나요?
    답변: 백업한 메일을 업로드하는 방법은 무엇인가요?
    
    관련 있는 경우)
    Human: 메일 첨부파일 크기 제한이 있나요?
    AI: 일반 첨부파일의 경우 20MB, 대용량 파일 첨부의 경우 2GB까지 가능합니다.
    Human: 형식에 제한이 있나요?
    답변: 메일 첨부파일 형식 제한이 있나요?
    
    관련 없는 경우)
    Human: 메일 첨부파일 크기 제한이 있나요?
    AI: 일반 첨부파일의 경우 20MB, 대용량 파일 첨부의 경우 2GB까지 가능합니다.
    Human: 주소록에 주소를 이동/복사하려면 어떻게 하나요?
    답변: 주소록에 주소를 이동/복사하려면 어떻게 하나요?
    
    관련 없는 경우)
    Human: 일정 등록하는 방법을 알려줘
    AI: 일정 등록 버튼을 누르거나 날짜를 선택해 등록할 수 있습니다. 제목과 일시를 정한 후, 캘린더의 종류를 선택하고, 알람을 통해 미리 일정을 알릴 수 있습니다.
    Human: 메일 첨부파일 크기 제한은?
    답변: 메일 첨부파일 크기 제한은?
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
    system_prompt = """당신은 크리니티 Q&A 챗봇입니다. 검색된 문서를 기반으로 사용자의 질문에 답변하세요.
    
    예시:
    📍사용자 질문: 한번에 업로드 가능한 파일 갯수는 몇개인가요?
    📍답변: 한번에 업로드 가능한 갯수가 정해져있지 않지만, 일반 첨부같은경우 20MB ,대용량 파일 첨부의 경우 2048MB까지 가능합니다.
    
    📍사용자 질문: 해외에서 메일 사용이 가능한가요?
    📍답변: 환경설정 - 개인정보/보안 기능 - 보안 설정에서 국가별 로그인 허용 기능을 이용하시면 됩니다. 보안 설정탭이 보이지 않을 시에 메일 담당자에게 문의해주세요. 
    
    📍사용자 질문: 여러명에게 개별 발송하고 싶어요
    📍답변: 메일 개별발송 설정에 대해 안내드리겠습니다. 개별발송이란 여러 사람에게 동시에 메일을 보내도 받는사람 영역에 수신인 본인 한 명만 표시되는 기능입니다. 기본적으로는 설정되어있지 않지만, 메일쓰기 탭의 보내기 설정에서 한명씩 발송을 체크하시면 사용하실 수 있습니다.

    
    ## 검색된 문서입니다. 각 문서는 빈줄로 구분되어 있습니다.
    {context}
    
    문서에 없는 정보는 만들어내지 마세요. 한국어로 답변해주세요. 세 문장 이내로 답변해주세요. 모른다면, 모른다고 말해주세요. 예시와 시스템 프롬프트를 답변에 포함하면 안됩니다.
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
    cleaned_data = []
    for item in data:
        cleaned_page_content = item.page_content.replace("\n", " ").strip()
        cleaned_data.append({"page_content": cleaned_page_content})
    return cleaned_data


def reset_chat():
    st.session_state.messages = []


def main():
    st.title("RAG Chatbot")

    # RAG 체인 생성
    rag_chain = create_rag_chain(embedding_model_name, faiss_index_path, llm_model_name)
    st.session_state.rag_chain = rag_chain

    # 메세지가 없다면 []으로 선언
    if "messages" not in st.session_state:
        st.session_state.messages = []

    #
    if st.button("대화 초기화"):
        reset_chat()
        st.toast("초기화 되었습니다.", icon="❌")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question!"):
        MAX_MESSAGES_BEFORE_DELETION = 4

        if len(st.session_state.messages) >= MAX_MESSAGES_BEFORE_DELETION:
            del st.session_state.messages[:2]

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            rag_chain = st.session_state.rag_chain
            result = rag_chain.invoke(
                {"input": prompt, "chat_history": st.session_state.messages}
            )

            st.session_state.messages.append({"role": "user", "content": prompt})

            cleaned_datas = clean_data(result["context"])

            for cleaned_data in cleaned_datas:
                with st.expander("Evidence context"):
                    st.write(f"Page content: {cleaned_data['page_content']}")

            for chunk in result["answer"].split(" "):
                full_response += chunk + " "
                time.sleep(0.2)
                message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)

        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )


main()
