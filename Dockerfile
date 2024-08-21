# Python 3.10 이미지 사용
FROM python:3.10

# 작업 디렉토리 설정
WORKDIR /app

# Python 가상환경 생성
RUN python -m venv /app/venv

ENV TZ=Asia/Seoul

# 가상환경 활성화 및 패키지 설치
COPY requirements.txt . 
RUN /app/venv/bin/pip install --upgrade pip && \
  /app/venv/bin/pip install -r requirements.txt

# 환경 변수 설정
ENV PATH="/app/venv/bin:$PATH"
ENV ENV="prod"

# 필요한 파일 및 디렉토리 복사
COPY .config /app/.config
COPY db /app/db
COPY agents /app/agents
COPY common /app/common
COPY pages /app/pages
COPY utils /app/utils
COPY Home.py .
COPY initializer.py .
COPY models.py .
COPY router.py .
COPY README.md .

# ollama 폴더 복사
COPY ollama /app/ollama

# ollama 폴더 내의 파일을 /app/venv/python3.10/langchain_ollama 폴더에 복사하여 덮어쓰기
RUN cp /app/ollama/chat_models.py /app/venv/lib/python3.10/site-packages/langchain_ollama && \
  cp /app/ollama/embeddings.py /app/venv/lib/python3.10/site-packages/langchain_ollama && \
  cp /app/ollama/llms.py /app/venv/lib/python3.10/site-packages/langchain_ollama


# Streamlit이 실행되는 포트 노출
EXPOSE 8501

# Streamlit 애플리케이션 실행 명령어
CMD ["streamlit", "run", "Home.py", "--server.port=8501", "--server.address=0.0.0.0"]