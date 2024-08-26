from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    FewShotPromptTemplate,
)
from models import FAQInput


class ManualQaAgent:
    """
    메뉴얼 QA 에이전트입니다.
    """

    def __init__(self, llm, retriever):
        self.llm = llm
        self.retriever = retriever

    def create_question_rephrasing_chain(self):
        system_prompt = """
        When you have previous conversation content and the latest user question, consider whether this question may be related to the previous conversation. 
        If so, rewrite the question so that it can be understood independently without knowing the prior conversation. 
        There is no need to answer the question; simply restructure it if necessary or leave it as is.

        Only respond with the restructured question.
        Make sure to provide your response in Korean.
        """

        examples = [
            {
                "chat_history": "Human: 메일 첨부파일 크기 제한이 있나요?\nAI: 일반 첨부파일의 경우 20MB, 대용량 파일 첨부의 경우 2GB까지 가능합니다.",
                "input": "형식에 제한이 있나요?",
                "rephrased": "메일 첨부파일 형식 제한이 있나요?",
            },
            {
                "chat_history": "Human: 일정 등록하는 방법을 알려줘\nAI: 일정 등록 버튼을 누르거나 날짜를 선택해 등록할 수 있습니다. 제목과 일시를 정한 후, 캘린더의 종류를 선택하고, 알람을 통해 미리 일정을 알릴 수 있습니다.",
                "input": "수정하는 방법은?",
                "rephrased": "일정 수정하는 방법은?",
            },
            {
                "chat_history": "Human: 주소록에 주소를 이동/복사하려면 어떻게 하나요?\nAI: 주소록에 주소를 이동/복사하려면, 주소록에서 주소를 선택한 후, 이동 또는 복사 버튼을 누르세요. 이동할 주소록을 선택하거나, 새로운 주소록을 만들어 이동할 수 있습니다.",
                "input": "복사하면 붙여넣기는 어떻게 하나요?",
                "rephrased": "주소록에 복사한 주소를 붙여넣는 방법은 무엇인가요?",
            },
            {
                "chat_history": "Human: 안녕\nAI: 안녕하세요! 무엇을 도와드릴까요?",
                "input": "메일 첨부파일 크기 제한이 있나요?",
                "rephrased": "메일 첨부파일 크기 제한이 있나요?",
            },
            {
                "chat_history": "Human: 일정 등록하는 방법을 알려줘 \nAI: 일정 등록 버튼을 누르거나 날짜를 선택해 등록할 수 있습니다. 제목과 일시를 정한 후, 캘린더의 종류를 선택하고, 알람을 통해 미리 일정을 알릴 수 있습니다.",
                "input": "메일 백업하는 방법을 알려줘",
                "rephrased": "메일 백업하는 방법을 알려줘",
            },
            {
                "chat_history": "Human: 메일 검색 기능에 대해 알고 싶어요.\nAI: 검색창에 검색어를 입력하면 전체, 보낸사람, 받는사람, 참조, 제목, 내용, 첨부파일명으로 메일을 검색할 수 있습니다.",
                "input": "제목으로 검색하는 방법은?",
                "rephrased": "메일 제목을 기준으로 검색하는 방법을 알려주세요.",
            },
        ]

        example_prompt = PromptTemplate.from_template(
            "#Previous conversation:\n{chat_history}\n#Latest user question:\n{input}\n#Rephrased question:\n{rephrased}"
        )

        few_shot_prompt = FewShotPromptTemplate(
            examples=examples,
            example_prompt=example_prompt,
            prefix=system_prompt,
            suffix="#Previous conversation:\n{chat_history}\n#Latest user question:\n{input}\n#Rephrased question:\n",
            input_variables=["chat_history", "input"],
        )

        return create_history_aware_retriever(self.llm, self.retriever, few_shot_prompt)

    def create_question_answering_chain(self):
        system_prompt = """
        You are 크리니티 Q&A chatbot. Could you please help by answering the user's questions based on the retrieved documents?
        Please try not to add any information that isn't in the documents.
        Also, it would be great if you could provide your responses in Korean.
        If you're unsure of an answer, it's perfectly okay to say '모르겠습니다'.
        
        <Retrieved_Documents>
        {context}
        </Retrieved_Documents>
        """

        prompt = ChatPromptTemplate.from_messages(
            [("system", system_prompt), ("human", "{input}\n")]
        )

        return create_stuff_documents_chain(self.llm, prompt)

    def answer_question(self, input, chat_history):
        formatted_chat_history = self.format_chat_history(chat_history)
        question_rephrasing_chain = self.create_question_rephrasing_chain()
        question_answering_chain = self.create_question_answering_chain()

        rag_chain = create_retrieval_chain(
            question_rephrasing_chain, question_answering_chain
        )

        if isinstance(input, str):
            input = FAQInput(question=input)

        question = input.question
        result = rag_chain.invoke(
            {"input": question, "chat_history": formatted_chat_history}
        )

        cleaned_datas = self.clean_data(result["context"])

        return result["answer"], cleaned_datas

    def clean_data(self, data):
        return [
            {"page_content": item.page_content.replace("\n", " ").strip()}
            for item in data
        ]

    def format_chat_history(self, messages):
        if not messages:
            return []
        return "\n".join(
            f"{'Human' if message['role'] == 'user' else 'AI'}: {message['content']}"
            for message in messages
        )
