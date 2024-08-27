from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from models import RouterResponse, TaskType
from langchain_core.messages import HumanMessage, SystemMessage


class Router:
    """라우터 에이전트

    사용자 입력에 따라 다른 에이전트를 호출합니다. 라우터 에이전트는 분류하는 역할만 수행합니다.

    에이전트 목록:
    - 번역(Translation)
    - 요약(Summary)
    - 문의(Inquiry)
    - 인사(Greeting)
    - 맞춤법&문맥(Proofread)
    - 기타(Others)
    """

    def __init__(
        self,
        llm,
        translation_agent,
        summarization_agent,
        qa_agent,
        email_assistant_agent,
        general_task_agent,
    ):
        self.llm = llm
        self.translation_agent = translation_agent
        self.summarization_agent = summarization_agent
        self.qa_agent = qa_agent
        self.email_assistant_agent = email_assistant_agent
        self.general_task_agent = general_task_agent

    def route(self, input):
        parser = JsonOutputParser(pydantic_object=RouterResponse)

        system_prompt = """You are a routing agent. Your task is to classify user input into one of the following task types:

        <Task Types>
        - Translation: Converting text from one language to another.
        - Summary: Summarizing long text into shorter form.
        - Inquiry: Answering questions or providing information related to email functions, settings, login processes, web folders, calendars, or other related features. This includes how to perform specific tasks, troubleshooting, or general inquiries about these functions.
        - Proofread: Checking spelling, grammar, and context issues in the email content.
        - TitleRecommendation : Suggesting a title based on the email content.
        - EmailTemplate: Generating email templates for different purposes.
        - Greeting: Greetings or introductions.
        - Others: Tasks that do not fall into the above categories.
        </Task Types>
        
        <Examples>
        - 안녕하세요 // Greeting
        - ㅎㅇ // Greeting
        - 너 누구야? // Greeting
        - 메일 첨부파일 크기 제한이 있나요? // Inquiry
        - 이미지를 첨부하려면 어떻게 하나요? // Inquiry
        - Mail delivery fails. An error message appears saying smtp sending failed. Please solve it. 번역해줘 // Translation
        - 메일 실패 문의에 대해 답변드립니다. 해당 문제는 SMTP 전송 실패로 인한 문제입니다. 이 글을 영어로 번역해주세요. // Translation
        - 현대 사회에서는 기술의 발전이 눈부시게 빠르게 이루어지고 있습니다. 이러한 기술 발전은 우리의 일상 생활에 많은 변화를 가져왔습니다. 이 글을 요약해줘 // Summary
        - 기술 발전이 가져오는 긍정적인 면만 있는 것은 아닙니다. 기술의 빠른 발전으로 인해 많은 직업이 자동화되고 있으며, 이에 따라 일자리의 감소와 같은 사회적 문제가 발생하고 있습니다. 또한, 개인정보 보호와 같은 문제도 대두되고 있습니다. 요약해줘. // Summary 
        - 이메일의 맞춤법을 검사해줘 // Proofread
        - 이 이메일의 문법이 맞는지 확인해줘 // Proofread
        - The club captain found the net in the 25th minute against Everton at Tottenham Hotspur Stadium in London when his team was leading 1-0. 이게 무슨 말이야? // Translation
        - 이 이메일의 제목을 추천해줘 // TitleRecommendation
        - 이 내용에 맞는 적절한 이메일 제목을 추천해줘 // TitleRecommendation
        - 이 내용을 바탕으로 이메일을 작성해줘 // EmailTemplate
        - 이 내용을 이메일 템플릿으로 만들어줘 // EmailTemplate
        </Examples>
        
        <Format>
        ```json
        {
            "task_type": "Translation" | "Summary" | "Inquiry" | "Proofread" | "EmailTemplate" | "Greeting" | "Others"
        }
        ```
        Json format with the key: `task_type`.
        </Format>
        """

        chat_prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=input),
            ]
        )

        chain = chat_prompt | self.llm | parser
        response = chain.invoke({"input": input})

        if "task_type" not in response:
            return TaskType.ETC

        return response["task_type"]

    def route_execute(self, input, chat_history):
        task_type = self.route(input)

        if task_type == TaskType.TRANSLATE:
            detect_language = self.translation_agent.detect_language(input)
            source_language = detect_language["source_language"]
            target_language = detect_language["target_language"]

            return self.translation_agent.translate(
                input,
                source_language,
                target_language,
            )
        elif task_type == TaskType.SUMMARIZE:
            return self.summarization_agent.summarize(input)
        elif task_type == TaskType.ManualQA:
            return self.qa_agent.answer_question(input, chat_history)
        elif task_type == TaskType.PROOFREAD:
            return self.email_assistant_agent.proofread(input)
        elif task_type == TaskType.TITLERECOMMENDATION:
            return self.email_assistant_agent.suggest_title(input)
        elif task_type == TaskType.EMAILTEMPLATE:
            return self.email_assistant_agent.generate_simple_email_template(input)
        elif task_type == TaskType.GREETING:
            return self.general_task_agent.greet(input)
        else:
            return self.general_task_agent.etc()
