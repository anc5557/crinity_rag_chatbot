from langchain_core.tools import tool
from src.chains import create_rag_chain


@tool
def rag_tool(query: str) -> str:
    """
    RAG Tool: 사용자 쿼리에 대한 정보를 검색하고 답변을 생성합니다.

    Parameters:
    - query (str): 사용자가 입력한 질문.

    Returns:
    - str: 검색된 정보에 기반한 응답.
    """
    # RAG 체인 생성
    rag_chain = create_rag_chain()

    # 사용자 쿼리에 대한 응답 생성
    result = rag_chain.invoke({"input": query})

    # 결과에서 생성된 응답 추출
    response = result.get("answer", "관련된 정보를 찾을 수 없습니다.")

    return response
