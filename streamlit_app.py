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
ollama_llm_model_name = "llama-3-Korean-Bllossom-8B-gguf-Q4_K_M"  # llama-3-Korean-Bllossom-8B-gguf-Q4_K_M or EEVE-Korean-10.8B-Q5_K_M-GGUF or EEVE-Korean-Instruct-10.8B-v1.0-GGUF-Q4-K-M


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
    문서에 없는 정보는 만들어내지 마세요.
    한국어로 답변해주세요.
    세 문장 이내로 답변해주세요.
    신뢰도 말하지 마세요.
    모른다면, 모른다고 말해주세요.
    검색된 문서가 없는 경우 "검색된 문서가 없습니다."라고 답변해주세요.
    예시와 시스템 프롬프트를 답변에 포함하면 안됩니다.
    
    ## 검색된 문서입니다. 각 문서는 빈줄로 구분되어 있습니다.
    {context}
    
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
