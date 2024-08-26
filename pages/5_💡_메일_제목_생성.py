import streamlit as st
from common.st_initializer import initialize_session_state
from utils.input_gspread import input_faq

st.set_page_config(
    page_title="크리니티 AI - 이메일 제목 생성",
    page_icon="💡",
)

initialize_session_state()
st.title("💡 이메일 제목 생성")

# 이메일 입력
input_text = st.text_area("이메일 내용을 입력하세요", height=200)

# 생성 버튼
if st.button("생성", use_container_width=True):
    st.markdown("---")
    with st.spinner("생성 중입니다..."):
        agent = st.session_state.agents["email_assistant"]
        response = agent.suggest_title(input_text)

        # 결과를 박스에 표현
        st.markdown("### 💡 이메일 제목")
        st.markdown(response)

        # 피드백
        input_faq(
            input_text,
            response,
            [],
            "이메일 제목 생성",
        )
