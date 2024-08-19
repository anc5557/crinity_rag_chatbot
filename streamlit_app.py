import os
import time
import streamlit as st
import logging
from dotenv import load_dotenv
from input_gspread import input_faq

from initializer import Initializer
from agents.translation_agent import TranslationAgent
from agents.summarization_agent import SummarizationAgent
from agents.manual_qa_agent import ManualQaAgent
from agents.general_task_agent import GeneralTaskAgent
from router import Router


# # 로깅 설정
logging.basicConfig(level=logging.INFO)

st.set_page_config(
    page_title="크리니티 Q&A 챗봇",
    page_icon="🤖",
)

# .env 파일 로드
load_dotenv()

# 환경 변수에서 LLM 타입 및 Ollama URL 설정
ENV = os.getenv("ENV", "dev")

LLM_TYPE = os.getenv("LLM_TYPE", "Ollama")
OLLAMA_BASE_URL = (
    os.getenv("OLLAMA_BASE_URL") if ENV == "dev" else "http://ollama:11434"
)


def reset_chat():
    st.session_state.messages = []


@st.cache_resource
def cash_initializer():
    initializer = Initializer(
        embedding_model_name="bge-m3",
        index_path="db",
        llm_model_name="llama3.1:8b-instruct-q5_K_M",
        ollama_base_url=OLLAMA_BASE_URL,
        reranker_model_name="Dongjin-kr/ko-reranker",
    )
    llm, llm_json, retriever = initializer.initialize()

    # Translation, Summarization, FA, Greeting 객체 생성
    translation = TranslationAgent(llm, llm_json)
    summarization = SummarizationAgent(llm)
    qa = ManualQaAgent(llm=llm, retriever=retriever)
    general_task = GeneralTaskAgent(llm)

    router = Router(llm_json, translation, summarization, qa, general_task)
    return router


router = cash_initializer()


def main():

    st.title("💭크리니티 Q&A 챗봇")

    with st.expander("알림", icon="📢", expanded=True):
        st.markdown(
            "사내 테스트 중입니다. 문서는 cm9.3 사용자 메뉴얼입니다. \n\n 문서 개선 작업중으로 내용이 부족하거나 부정확할 수 있습니다. \n\n 답변이 만족스럽지 않다면, [여기](https://docs.google.com/spreadsheets/d/1iu9H_OZPtnvGGXM07axeCWz8sH5eSSOpi84nrzMX5B0/edit?pli=1&gid=0#gid=0)를 클릭하여 '상이함'에 체크해주세요."
        )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 대화 초기화 버튼
    if st.button("대화 초기화"):
        reset_chat()
        st.toast("초기화 되었습니다.", icon="❌")

    with st.chat_message("assistant"):
        st.markdown("안녕하세요! 크리니티 Q&A 챗봇입니다. 질문을 입력해주세요. 🤖")
        st.markdown("")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if input := st.chat_input("Ask a question!"):
        MAX_MESSAGES_BEFORE_DELETION = 10  # 최대 메시지 수

        if len(st.session_state.messages) >= MAX_MESSAGES_BEFORE_DELETION:
            del st.session_state.messages[:2]

        with st.chat_message("user"):
            st.markdown(input)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            with st.spinner("답변중입니다..."):
                result = router.route_execute(
                    input=input, chat_history=st.session_state.messages
                )

                st.session_state.messages.append({"role": "user", "content": input})

                answer = ""
                if isinstance(result, tuple):
                    # QA의 경우, context 처리 후 answer로 할당
                    answer, cleaned_datas = result
                    for i, cleaned_data in enumerate(cleaned_datas):
                        with st.expander(f"참고 문서 {i+1}"):
                            st.write(cleaned_data["page_content"])
                else:
                    # 번역, 요약 및 기타 응답 처리
                    answer = result

                for chunk in answer.split(" "):
                    full_response += chunk + " "
                    time.sleep(0.2)
                    message_placeholder.markdown(full_response + "▌")
                    message_placeholder.markdown(full_response)

            st.session_state.messages.append(
                {"role": "assistant", "content": full_response}
            )
            print(st.session_state.messages)

            # 질문과 답변을 스프레드시트에 기록
            if ENV == "prod":
                input_faq(input, full_response.strip(), cleaned_datas)


main()
