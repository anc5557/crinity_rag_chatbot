from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


class GeneralTaskAgent:
    """
    기타 작업 에이전트입니다.
    """

    def __init__(self, llm):
        self.llm = llm

    def greet(self, input):
        """인사에 대한 답변을 반환합니다."""

        system_prompt = """
        Could you please act as a 크리니티 Q&A chatbot? When a user greets you, could you respond to their greeting and provide a brief introduction in Korean?

        Here's some information you might find helpful for the introduction:

        <크리니티 Introduction>
        - 크리니티 provides webmail solutions.
        - We are the No.1 webmail company.
        - Established in 1998, 크리니티 has over 20 years of experience in email collaboration and security, offering top-notch technology to ensure safe communication for our customers.
        </크리니티 Introduction>
        """

        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=input),
            ]
        )

        chain = prompt | self.llm | StrOutputParser()

        response = chain.invoke({"input": input})
        return response

    def etc(self):
        """답변할 수 없는 경우"""
        return "답변할 수 없습니다."
