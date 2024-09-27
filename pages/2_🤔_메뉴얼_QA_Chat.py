import streamlit as st
import time

from utils.input_gspread import input_faq
from common.st_initializer import initialize_session_state

st.set_page_config(
    page_title="크리니티 AI - 메뉴얼",
    page_icon="🤔",
)
initialize_session_state()


st.title("🤔 메뉴얼 QA Chat")

with st.expander("알림", icon="📢", expanded=True):
    st.markdown(
        "사내 테스트 중입니다. CM9.3 USER 가이드 문서에 해당하는 질문에 답변합니다.\n\n  답변이 만족스럽지 않다면, [여기](https://docs.google.com/spreadsheets/d/1iu9H_OZPtnvGGXM07axeCWz8sH5eSSOpi84nrzMX5B0/edit?pli=1&gid=0#gid=0)를 클릭하여 '상이함'에 체크해주세요. \n\n 문서 내용 또는 챗봇 정보가 궁금하시다면 [여기](https://docs.google.com/spreadsheets/d/1iu9H_OZPtnvGGXM07axeCWz8sH5eSSOpi84nrzMX5B0/edit?pli=1&gid=481801826#gid=481801826)를 클릭해주세요."
    )


def reset_chat():
    st.session_state.messages = []


if "messages" not in st.session_state:
    st.session_state.messages = []

# 대화 초기화 버튼
if st.button("대화 초기화"):
    reset_chat()
    st.toast("초기화 되었습니다.", icon="❌")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if input := st.chat_input("질문을 입력하세요!"):
    MAX_MESSAGES_BEFORE_DELETION = 10  # 최대 메시지 수

    if len(st.session_state.messages) >= MAX_MESSAGES_BEFORE_DELETION:
        del st.session_state.messages[:2]

    with st.chat_message("user"):
        st.markdown(input)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        with st.spinner("답변중입니다..."):
            result, cleaned_datas = st.session_state.agents["qa"].answer_question(
                input=input, chat_history=st.session_state.messages
            )

            st.session_state.messages.append({"role": "user", "content": input})

            answer = result

            for i, cleaned_data in enumerate(cleaned_datas):
                with st.expander(f"참고 문서 {i+1}"):
                    st.write(cleaned_data["page_content"])

            for chunk in answer.split(" "):
                full_response += chunk + " "
                time.sleep(0.2)
                message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)

        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )

        # 질문과 답변을 스프레드시트에 기록
        input_faq(input, full_response.strip(), cleaned_datas, "메뉴얼")
