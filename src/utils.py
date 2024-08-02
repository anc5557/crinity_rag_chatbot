import streamlit as st


def clean_data(data):
    """데이터 정제"""
    cleaned_data = []
    for item in data:
        cleaned_page_content = item.page_content.replace("\n", " ").strip()
        cleaned_data.append({"page_content": cleaned_page_content})
    return cleaned_data


def format_chat_history(messages):
    """대화 내용을 텍스트 형식으로 변환"""
    formatted_messages = []
    for message in messages:
        if message["role"] == "user":
            formatted_messages.append(f"Human: {message['content']}")
        elif message["role"] == "assistant":
            formatted_messages.append(f"AI: {message['content']}")
    return "\n".join(formatted_messages)


def reset_chat():
    st.session_state.messages = []
