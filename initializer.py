from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder


class Initializer:
    """
    LLM 및 검색 도구 초기화를 위한 클래스입니다.
    """

    def __init__(
        self,
        embedding_model_name,
        index_path,
        llm_model_name,
        ollama_base_url,
        reranker_model_name,
    ):
        self.embedding_model_name = embedding_model_name
        self.index_path = index_path
        self.llm_model_name = llm_model_name
        self.ollama_base_url = ollama_base_url
        self.reranker_model_name = reranker_model_name

    def initialize_embedding_model(self):
        """임베딩 모델을 초기화합니다."""
        return OllamaEmbeddings(
            model=self.embedding_model_name,
            base_url=self.ollama_base_url,
        )

    def initialize_vectorstore(self, embedding_model):
        """FAISS 벡터스토어를 초기화합니다."""
        return FAISS.load_local(
            folder_path=self.index_path,
            embeddings=embedding_model,
            allow_dangerous_deserialization=True,
        )

    def initialize_chat_ollama_json(self):
        """ChatOllama를 초기화합니다."""
        return ChatOllama(
            model=self.llm_model_name,
            base_url=self.ollama_base_url,
            format="json",
            temperature=0.3,
        )

    def initialize_chat_ollama(self):
        """ChatOllama를 초기화합니다."""
        return ChatOllama(
            model=self.llm_model_name,
            base_url=self.ollama_base_url,
            temperature=0.3,
        )

    def initialize_reranker(self):
        """리랭커 모델을 초기화합니다."""
        cross_encoder = HuggingFaceCrossEncoder(model_name=self.reranker_model_name)
        return CrossEncoderReranker(model=cross_encoder, top_n=3)

    def initialize_retriever(self, vectorstore):
        """리트리버를 초기화합니다."""
        return vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": 3}
        )

    def initialize(self):
        """모든 도구를 초기화하여 반환합니다."""
        embedding_model = self.initialize_embedding_model()
        vectorstore = self.initialize_vectorstore(embedding_model)
        retriever = self.initialize_retriever(vectorstore)
        llm = self.initialize_chat_ollama()
        llm_json = self.initialize_chat_ollama_json()
        # reranker = self.initialize_reranker()
        return llm, llm_json, retriever
