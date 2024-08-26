import streamlit as st
from common.st_initializer import initialize_session_state
from utils.input_gspread import input_faq

st.set_page_config(page_title="크리니티 AI - 메일 템플릿", page_icon="✉️")

initialize_session_state()

st.title("✉️  메일 템플릿")


# 추가 입력 필드
target = st.selectbox("수신 대상", options=["고객사", "파트너", "동료", "상사"])
tone = st.selectbox(
    "어투", options=["정중한", "전문적인", "자신감", "친숙한", "캐주얼한", "공식적인"]
)
category = st.selectbox("상황", options=["공지/안내", "인사", "사과"])

# 이메일 입력
input_text = st.text_area(
    "메일에 들어가야 할 내용을 구체적으로 작성해 주세요.", height=200
)
# 생성 버튼
if st.button("생성", use_container_width=True):
    st.markdown("---")
    with st.spinner("생성 중입니다..."):
        agent = st.session_state.agents["email_assistant"]
        response = agent.generate_email_template(
            content=input_text, target=target, tone=tone, category=category
        )

        # 결과를 박스에 표현
        st.markdown("### ✉️ 메일 템플릿")
        st.markdown(response)

        # 피드백
        input_faq(
            input_text,
            response,
            [],
            "메일 템플릿",
            "수신 대상: {}, 어투: {}, 상황: {}".format(target, tone, category),
        )
