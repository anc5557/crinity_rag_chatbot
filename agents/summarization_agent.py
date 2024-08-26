from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


class SummarizationAgent:
    """
    요약 에이전트입니다.
    """

    def __init__(self, llm):
        self.llm = llm

    def summarize(self, text_to_summarize: str) -> str:
        system_prompt = """"
        You are a summarization expert. Summarize the given text.
        """
        chat_prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=text_to_summarize),
            ]
        )

        chain = chat_prompt | self.llm | StrOutputParser()
        response = chain.invoke({"text_to_summarize": text_to_summarize})
        return response
