import streamlit as st
from common.st_initializer import initialize_session_state

st.set_page_config(
    page_title="크리니티 AI - 메일 맞춤법 및 맥락 검사",
    page_icon="🔍",
)

initialize_session_state()

st.title("🔍 메일 맞춤법 및 맥락 검사")

# 이메일 내용 입력
input_text = st.text_area("검사할 메일 내용을 입력하세요", height=200)

# 검사 버튼
if st.button("검사", use_container_width=True):
    st.markdown("---")
    with st.spinner("검사 중입니다! 잠시만 기다려주세요..."):
        agent = st.session_state.agents["email_assistant"]
        response = agent.proofread(input_text)

        # 결과를 박스에 표현
        st.markdown("### 📝 검사 결과")
        st.markdown(response)
