import streamlit as st
from common.st_router import get_router
import time

router = get_router()


def reset_chat():
    st.session_state.messages = []


st.title("💬 크리니티 챗봇")

with st.expander("알림", icon="📢", expanded=True):
    st.markdown(
        "사내 테스트 중입니다. 문서는 cm9.3 사용자 메뉴얼입니다. \n\n 문서 개선 작업중으로 내용이 부족하거나 부정확할 수 있습니다. \n\n 답변이 만족스럽지 않다면, [여기](https://docs.google.com/spreadsheets/d/1iu9H_OZPtnvGGXM07axeCWz8sH5eSSOpi84nrzMX5B0/edit?pli=1&gid=0#gid=0)를 클릭하여 '상이함'에 체크해주세요."
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

if st.button("대화 초기화"):
    reset_chat()
    st.toast("초기화 되었습니다.", icon="❌")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if input := st.chat_input("Ask a question!"):
    MAX_MESSAGES_BEFORE_DELETION = 10

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

            answer = result if isinstance(result, str) else result[0]

            for chunk in answer.split(" "):
                full_response += chunk + " "
                time.sleep(0.2)
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )
