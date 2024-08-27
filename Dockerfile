# Python 3.10 이미지 사용
FROM python:3.10

# 작업 디렉토리 설정
WORKDIR /app

# Python 가상환경 생성
RUN python -m venv /app/venv

# 타임존 설정
ENV TZ=Asia/Seoul

# torch와 faiss-gpu 먼저 설치
RUN /app/venv/bin/pip install --upgrade pip && \
  /app/venv/bin/pip install torch==2.3.1 faiss-gpu

# Python 가상환경 활성화 및 패키지 설치
COPY server_requirements.txt .
RUN /app/venv/bin/pip install --default-timeout=100 -r server_requirements.txt

# 환경 변수 설정
ENV PATH="/app/venv/bin:$PATH"
ENV ENV="prod"

# 필요한 파일 및 디렉토리 복사
COPY . /app/

# Streamlit이 실행되는 포트 노출
EXPOSE 8501

# Streamlit 애플리케이션 실행 명령어
CMD ["streamlit", "run", "Home.py", "--server.port=8501", "--server.address=0.0.0.0"]