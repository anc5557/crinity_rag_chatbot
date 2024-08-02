import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# 환경 변수에서 LLM 타입 및 Ollama URL 설정
ENV = os.getenv("ENV", "dev")
LLM_TYPE = os.getenv("LLM_TYPE", "Ollama")
OLLAMA_BASE_URL = (
    os.getenv("OLLAMA_BASE_URL") if ENV == "dev" else "http://ollama:11434"
)

# 모델 및 인덱스 경로 설정
faiss_index_path = "db"
embedding_model_name = "BAAI/bge-m3"
huggingface_llm_model_name = "beomi/gemma-ko-2b"
ollama_llm_model_name = "llama3.1:8b-instruct-q5_K_M"
reranker_model_name = "Dongjin-kr/ko-reranker"
