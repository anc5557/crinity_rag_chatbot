import os
import argparse
import logging
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# 로깅 설정
logging.basicConfig(level=logging.INFO)

# .env 파일 로드
load_dotenv()

faiss_index_path = "db"
embedding_model_name = "BAAI/bge-m3"  # "jhgan/ko-sroberta-multitask"


def load_embedding_model(model_name):
    """임베딩 모델 로드"""
    embedding_model = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return embedding_model


def load_vectorstore(index_path, _embedding_model):
    """벡터스토어 로드"""
    vectorstore = FAISS.load_local(
        folder_path=index_path,
        embeddings=_embedding_model,
        allow_dangerous_deserialization=True,
    )
    return vectorstore


def main(question):
    # 임베딩 모델 및 벡터스토어 로드
    embedding_model = load_embedding_model(embedding_model_name)
    vectorstore = load_vectorstore(faiss_index_path, embedding_model)

    # 벡터스토어를 이용한 검색 수행
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 2, "fetch_k": 20},
        verbose=True,
    )

    search_results = retriever.invoke(question)

    # 검색된 결과 출력
    if search_results:
        for i, (document) in enumerate(search_results):
            print(f"결과 {i+1}:")
            print(f"{document.page_content}")
            # print(f"점수: {score}")
            print("-" * 80)
    else:
        print("검색된 문서가 없습니다.")


if __name__ == "__main__":
    question = "메일 백업하는 방법 알려줘"
    main(question)
