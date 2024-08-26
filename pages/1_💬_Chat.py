import streamlit as st
import time
import os

from utils.input_gspread import input_faq
from common.st_initializer import initialize_session_state

st.set_page_config(
    page_title="크리니티 챗봇",
    page_icon="🤖",
)

initialize_session_state()


def reset_chat():
    st.session_state.messages = []


st.title("💭크리니티 Q&A 챗봇")

with st.expander("알림", icon="📢", expanded=True):
    st.markdown(
        "사내 테스트 중입니다. 문서는 cm9.3 사용자 메뉴얼입니다. \n\n  답변이 만족스럽지 않다면, [여기](https://docs.google.com/spreadsheets/d/1iu9H_OZPtnvGGXM07axeCWz8sH5eSSOpi84nrzMX5B0/edit?pli=1&gid=0#gid=0)를 클릭하여 '상이함'에 체크해주세요."
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
            result = st.session_state.agents["router"].route_execute(
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

        # 질문과 답변을 스프레드시트에 기록

        if "cleaned_datas" not in locals():
            cleaned_datas = []
        input_faq(input, full_response.strip(), cleaned_datas, "챗봇")
