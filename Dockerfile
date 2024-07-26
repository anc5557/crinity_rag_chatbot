# Python 3.10 이미지 사용
FROM python:3.10

# 작업 디렉토리 설정
WORKDIR /app

# Python 가상환경 생성
RUN python -m venv /app/venv

# 가상환경 활성화 및 패키지 설치
COPY requirements.txt .
RUN /app/venv/bin/pip install --upgrade pip && \
  /app/venv/bin/pip install -r requirements.txt

# 환경 변수 설정
ENV PATH="/app/venv/bin:$PATH"
ENV LLM_TYPE="Ollama"
ENV OLLAMA_BASE_URL="http://ollama:11434"

# Streamlit 애플리케이션 코드 복사
COPY streamlit_app.py .

# 데이터베이스 파일 복사
COPY db /app/db

# Streamlit이 실행되는 포트 노출
EXPOSE 8501

# Streamlit 애플리케이션 실행 명령어
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
