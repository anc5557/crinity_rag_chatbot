from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


class EmailAssistantAgent:
    """
    이메일 내용의 맥락 및 맞춤법 검사, 제목 추천, 템플릿(안내문, 공지, 사과 ...) 생성 등의 기능을 수행하는 에이전트입니다.
    """

    def __init__(self, llm):
        self.llm = llm

    def proofread(self, input: str):
        """이메일 내용의 맞춤법, 맥락 검사를 수행합니다.

        Args:
            input (str): 이메일 내용
        """

        chat_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "당신은 이메일 내용의 맥락 및 맞춤법 검사 전문가입니다.",
                ),
                (
                    "user",
                    f"""
                    가이드에 따라 다음 이메일 내용을 파악하고 수정해주세요.

                    <가이드>
                    1. 맞춤법 및 문법 오류 식별
                    2. 자연스럽고 논리적인 문장인지 확인
                    3. 전체 맥락 적합성 검토
                    4. 이를 바탕으로 수정 제안
                    </가이드>
                    
                    <이메일 내용>
                    {input}
                    </이메일 내용>

                    Let's go step by step.
                    """,
                ),
            ]
        )

        chain = chat_prompt | self.llm | StrOutputParser()

        response = chain.invoke({"input": input})
        return response

    def suggest_title(self, input: str):
        """이메일 내용을 기반으로 제목을 추천합니다.

        Args:
            input (str): 이메일 내용
        """

        chat_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "당신은 이메일 제목 추천 전문가입니다. 이메일 내용을 기반으로 적절한 제목을 추천해주세요. ",
                ),
                (
                    "user",
                    f"""
                    다음 이메일 내용을 가이드에 따라 적절한 제목을 추천해주세요.

                    <가이드>
                    1. 이메일의 중심 내용과 목적을 찾습니다.
                    2. 중요한 키워드를 파악합니다.
                    3. 위 내용을 바탕으로 간결하고 명확하게 제목을 작성하고 추천합니다.
                    </가이드>
                    
                    <이메일 내용>
                    {input}
                    </이메일 내용>
                    
                    Let's go step by step.
                    """,
                ),
            ]
        )

        chain = chat_prompt | self.llm | StrOutputParser()

        response = chain.invoke({"input": input})
        return response

    def generate_email_template(
        self,
        content: str,
        target: str = "고객사",
        tone: str = "정중한",
        category: str = "공지/안내",
    ):
        """이메일 템플릿을 생성합니다.

        Args:
            content (str): 사용자가 요청한 이메일에 포함될 내용
            target (str): 이메일 수신 대상
            tone (str): 이메일 어투 (예: 정중한, 캐주얼한)
            category (str): 이메일 카테고리 (예: 공지/안내, 인사, 사과)
        """

        chat_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    f"당신은 이메일 템플릿 생성 전문가입니다. {target}에게 {tone} 어투로 {category} 이메일을 작성해주세요.",
                ),
                (
                    "user",
                    """
                    가이드에 따라 다음 이메일 템플릿을 작성해주세요.
                    
                    <포함할 내용>
                    {content}
                    </포함할 내용>
                    
                    <가이드>
                    1. 이메일 수신 대상, 어투, 카테고리 파악합니다.
                    2. 포함할 내용을 분석합니다.
                    3. 이를 바탕으로 적절한 이메일 템플릿을 작성합니다. 
                    </가이드>
                    
                    Let's go step by step.
                    """,
                ),
            ]
        )

        chain = chat_prompt | self.llm | StrOutputParser()

        response = chain.invoke({"content": content})
        return response

    def generate_simple_email_template(self, input: str):
        """단순한 이메일 템플릿을 생성합니다.

        Args:
            input (str): 사용자 질문
        """

        chat_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "당신은 이메일 템플릿 생성 전문가입니다. 사용자의 질문을 바탕으로 이메일 템플릿을 작성해주세요.",
                ),
                (
                    "user",
                    """
                    가이드에 따라 다음 이메일 템플릿을 작성해주세요.
                    
                    <질문>
                    {input}
                    </질문>
                    
                    <가이드>
                    1. 사용자의 질문에서 요청 사항을 파악합니다.(대상, 말투, 목적 등) 
                    2. 중요한 내용을 파악합니다.
                    3. 키워드를 파악합니다.
                    4. 이를 바탕으로 이메일 템플릿을 작성합니다.
                    </가이드>
                    
                    Let's go step by step.
                    """,
                ),
            ]
        )

        chain = chat_prompt | self.llm | StrOutputParser()

        response = chain.invoke({"input": input})
        return response
