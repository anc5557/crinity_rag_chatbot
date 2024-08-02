# src/models.py
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
import config.config as config
import streamlit as st


@st.cache_resource
def load_embedding_model(model_name):
    """임베딩 모델 로드"""
    return HuggingFaceEmbeddings(
        model_name=model_name, encode_kwargs={"normalize_embeddings": True}
    )


@st.cache_resource
def load_vectorstore(index_path, _embedding_model):
    """벡터스토어 로드"""
    return FAISS.load_local(
        folder_path=index_path,
        embeddings=_embedding_model,
        allow_dangerous_deserialization=True,
    )


@st.cache_resource
def load_llm(llm_model_name, llm_type):
    """LLM 로드"""
    if llm_type == "HuggingFace":
        return HuggingFacePipeline.from_model_id(
            model_id=llm_model_name,
            device=0,
            task="text-generation",
            pipeline_kwargs={
                "max_length": 1000,
                "num_return_sequences": 1,
                "max_new_tokens": 500,
            },
        )
    elif llm_type == "Ollama":
        return Ollama(model=llm_model_name, base_url=config.OLLAMA_BASE_URL)
