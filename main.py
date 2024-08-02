import streamlit as st
import time
from chains import create_rag_chain
from scripts.input_gspread import input_faq
from utils import clean_data, format_chat_history, reset_chat
import config.config as config
from config.logging_config import *


def main():
    st.set_page_config(
        page_title="크리니티 Q&A 챗봇",
        page_icon="🤖",
    )

    st.title("💭크리니티 Q&A 챗봇")

    with st.expander("알림", icon="📢", expanded=True):
        """사내 테스트 중입니다. 문서는 cm9.3 사용자 메뉴얼입니다. \n\n 문서 개선 작업중으로 내용이 부족하거나 부정확할 수 있습니다. \n\n 답변이 만족스럽지 않다면, [여기](https://docs.google.com/spreadsheets/d/1iu9H_OZPtnvGGXM07axeCWz8sH5eSSOpi84nrzMX5B0/edit?pli=1&gid=0#gid=0)를 클릭하여 '상이함'에 체크해주세요."""

    if config.LLM_TYPE == "HuggingFace":
        llm_model_name = config.huggingface_llm_model_name
    else:
        llm_model_name = config.ollama_llm_model_name

    # RAG 체인 생성
    rag_chain = create_rag_chain()
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
                formatted_chat_history = format_chat_history(st.session_state.messages)
                result = rag_chain.invoke(
                    {"input": prompt, "chat_history": formatted_chat_history}
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
            if config.ENV == "prod":
                input_faq(prompt, full_response.strip(), cleaned_datas)


main()
