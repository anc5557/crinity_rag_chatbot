# crinity_rag_chatbot

메뉴얼 문서를 임베딩하여 사용자의 질문에 대답하는 챗봇입니다.
메뉴얼 문서는 CM9.3의 메뉴얼을 사용하였습니다. 문서에 부족한 부분이 있어 보충이 필요합니다.

## Ollama

<https://huggingface.co/teddylee777/EEVE-Korean-Instruct-10.8B-v1.0-gguf>

ollama 모델을 사용하였습니다.  
위 링크에서 모델을 다운로드 받아  
터미널에서 "ollama create <모델이름> -f <Modelfile 경로>"를 실행하여 모델을 생성합니다.  

## Requirements

streamlit==1.36.0  
langchain==0.2.11  
langchain-community==0.2.10  
langchain_huggingface==0.0.3  
torch==2.4.0  
faiss-gpu  
python-dotenv  
gspread  

## create_faiss_index.py

메뉴얼 문서를 임베딩해 로컬에 faiss index를 저장하는 코드입니다.  

- 실행 방법  
  - python create_faiss_index.py --json_path [JSON 파일 경로] --output_path [FAISS 인덱스 저장 경로]
  - output_path의 기본값은 db/ 입니다. 기본값을 사용하는 것을 권장합니다.

## cuda_check.py

cuda가 사용 가능한지 확인하는 코드입니다.

## streamlit_app.py

streamlit을 사용하여 챗봇을 실행하는 코드입니다.  
streamlit run streamlit_app.py 명령어로 실행합니다.

## .env

메뉴얼 문서와 스프레드 시트 API를 사용하기 위한 환경 변수 파일은 비공개로 관리합니다.
