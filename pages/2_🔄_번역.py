import streamlit as st
from common.st_initializer import initialize_session_state
from utils.input_gspread import input_faq

st.set_page_config(
    page_title="크리니티 AI - 번역",
    page_icon="🔄",
)

initialize_session_state()

st.title("🔄 번역")

# 언어 선택
language_options = ["English", "Korean", "Japanese", "Chinese"]
source_language = st.selectbox("Source Language", language_options, index=0)
target_language = st.selectbox("Target Language", language_options, index=1)


# 번역할 문장 입력
input_text = st.text_area("Enter the text to translate", height=200)


# 번역 버튼
if st.button("번역", use_container_width=True):
    st.markdown("---")
    with st.spinner("번역중입니다..."):
        agent = st.session_state.agents["translation"]
        translated_text = agent.translate(input_text, source_language, target_language)

        # 결과를 박스에 표현
        st.markdown("### 🌏 번역 결과")
        st.markdown(translated_text)

        # # 피드백
        # input_faq(
        #     input_text,
        #     translated_text,
        #     [],
        #     "번역",
        #     f"{source_language} -> {target_language}",
        # )
