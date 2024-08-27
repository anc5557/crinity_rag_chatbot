import streamlit as st
from common.st_initializer import initialize_session_state
from utils.input_gspread import input_faq

st.set_page_config(
    page_title="크리니티 AI - 요약",
    page_icon="📄",
)

initialize_session_state()
st.title("📄 요약")

# 요약할 문장 입력
input_text = st.text_area("Enter the text to summarize", height=200)

# 요약 버튼
if st.button("요약", use_container_width=True):
    st.markdown("---")
    with st.spinner("요약중입니다..."):
        agent = st.session_state.agents["summarization"]
        summarized_text = agent.summarize(input_text)

        # 결과를 박스에 표현
        st.markdown("### 📄 요약 결과")
        st.markdown(summarized_text)

        # 피드백
        input_faq(
            input_text,
            summarized_text,
            [],
            "요약",
        )
