import streamlit as st
from common.st_initializer import initialize_session_state

st.set_page_config(
    page_title="크리니티 AI",
    page_icon="🤖",
)

initialize_session_state()

st.write("# 환영합니다! 크리니티 AI입니다.")
st.markdown("사이드바에서 원하는 기능을 선택하여 사용해 주세요.")
st.markdown(
    "피드백은 언제나 환영입니다. [여기](https://docs.google.com/spreadsheets/d/1iu9H_OZPtnvGGXM07axeCWz8sH5eSSOpi84nrzMX5B0/edit?pli=1&gid=481801826#gid=481801826)를 클릭하여 피드백을 남겨주세요."
)

st.markdown(
    """
    ### 📚 기능 목록
    1. Chat
        - 채팅을 통해 AI와 대화할 수 있습니다.
        - 라우터 방식으로 아래 다른 기능을 통합하여 사용할 수 있습니다.
        - 메뉴얼에 대한 질문을 입력하면, 문서를 참고하여 답변을 제공합니다.
        - 예시) ~~를 요약해줘, ~~를 번역해줘, ~~ 맞춤법 ...
    2. 번역
    3. 요약
    4. 메일 맞춤법 및 맥락 검사
    5. 메일 제목 생성
    6. 메일 템플릿
    7. 지식공유DB 바탕으로 답변받기(예정)
    8. SQL 쿼리 생성기(예정)
    
    """
)
