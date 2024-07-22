import time
import streamlit as st
import logging
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
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


# LLM 타입 설정
LLM_TYPE = "Ollama"  # "HuggingFace" 또는 "Ollama"

faiss_index_path = "db"
embedding_model_name = "jhgan/ko-sroberta-multitask"
huggingface_llm_model_name = "beomi/gemma-ko-2b"  # beomi/gemma-ko-2b
ollama_llm_model_name = "EEVE-Korean-Instruct-10.8B-v1.0-GGUF-Q4-K-M"  # llama-3-Korean-Bllossom-8B-gguf-Q4_K_M or EEVE-Korean-10.8B-Q5_K_M-GGUF or EEVE-Korean-Instruct-10.8B-v1.0-GGUF-Q4-K-M


@st.cache_resource
def load_embedding_model(model_name):
    embedding_model = HuggingFaceEmbeddings(
        model_name=model_name, encode_kwargs={"normalize_embeddings": True}
    )
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
def load_llm(llm_model_name, llm_type):
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
        llm = Ollama(model=llm_model_name)
    return llm


@st.cache_resource
def create_rag_chain(embedding_model_name, faiss_index_path, llm_model_name, llm_type):
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
    system_prompt = """
    당신은 질문 재구성자입니다. 이전 대화 내용과 최신 사용자 질문이 있을 때, 이 질문이 이전 대화 내용과 관련이 있을 수 있습니다.
    관련이 있는 경우, 이전 대화 내용을 참고하여 사용자의 최신 질문을 재구성하세요. 
    관련 없는 경우, 사용자 질문을 그대로 반복해주세요.
    
    예시:
    관련 있는 경우)
    1.
    Human: 메일 첨부파일 크기 제한이 있나요?
    AI: 일반 첨부파일의 경우 20MB, 대용량 파일 첨부의 경우 2GB까지 가능합니다.
    Human: 형식에 제한이 있나요?
    답변: 메일 첨부파일 형식 제한이 있나요?
    
    2.
    Human: 일정 등록하는 방법을 알려줘
    AI: 일정 등록 버튼을 누르거나 날짜를 선택해 등록할 수 있습니다. 제목과 일시를 정한 후, 캘린더의 종류를 선택하고, 알람을 통해 미리 일정을 알릴 수 있습니다.
    Human: 수정하는 방법은?
    답변: 일정 수정하는 방법은?
    
    3.
    Human: 주소록에 주소를 이동/복사하려면 어떻게 하나요?
    AI: 주소록에 주소를 이동/복사하려면, 주소록에서 주소를 선택한 후, 이동 또는 복사 버튼을 누르세요. 이동할 주소록을 선택하거나, 새로운 주소록을 만들어 이동할 수 있습니다.
    Human: 복사하면 붙여넣기는 어떻게 하나요?
    답변: 주소록에 복사한 주소를 붙여넣는 방법은 무엇인가요?
    
    관련 없는 경우)
    1.
    Human: 메일 첨부파일 크기 제한이 있나요?
    AI: 일반 첨부파일의 경우 20MB, 대용량 파일 첨부의 경우 2GB까지 가능합니다.
    Human: 주소록에 주소를 이동/복사하려면 어떻게 하나요?
    답변: 주소록에 주소를 이동/복사하려면 어떻게 하나요?
    
    2.
    Human: 일정 등록하는 방법을 알려줘
    AI: 일정 등록 버튼을 누르거나 날짜를 선택해 등록할 수 있습니다. 제목과 일시를 정한 후, 캘린더의 종류를 선택하고, 알람을 통해 미리 일정을 알릴 수 있습니다.
    Human: 메일 첨부파일 크기 제한은?
    답변: 메일 첨부파일 크기 제한은?
    
    3.
    Human: 안녕
    AI: 안녕하세요! 무엇을 도와드릴까요?
    Human: 메일 첨부파일 크기 제한이 있나요?
    답변: 메일 첨부파일 크기 제한이 있나요?
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
    문서에 없는 정보는 만들어내지 마세요.
    한국어로 답변해주세요.
    세 문장 이내로 답변해주세요.
    모른다면, 모른다고 말해주세요.
    검색된 문서가 없는 경우 "검색된 문서가 없습니다."라고 답변해주세요.
    
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
    cleaned_data = []
    for item in data:
        cleaned_page_content = item.page_content.replace("\n", " ").strip()
        cleaned_data.append({"page_content": cleaned_page_content})
    return cleaned_data


def reset_chat():
    st.session_state.messages = []


def main():
    st.title("크리니티 Q&A 챗봇")

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

    #
    if st.button("대화 초기화"):
        reset_chat()
        st.toast("초기화 되었습니다.", icon="❌")

    with st.chat_message("assistant"):
        st.markdown("안녕하세요! 크리니티 Q&A 챗봇입니다. 질문을 입력해주세요. 🤖")

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

            with st.spinner("답변중입니다..."):
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
