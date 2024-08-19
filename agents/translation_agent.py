from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain.output_parsers.fix import OutputFixingParser
from langchain_core.prompts import ChatPromptTemplate
from models import DetectTranslationRequest


class TranslationAgent:
    """
    번역 에이전트입니다.
    """

    def __init__(self, llm, llm_json):
        self.llm = llm
        self.llm_json = llm_json

    def detect_language(self, input: str) -> dict:
        parser = JsonOutputParser(pydantic_object=DetectTranslationRequest)
        fixing_parser = OutputFixingParser.from_llm(llm=self.llm_json, parser=parser)

        system_prompt = """
        Before translation, detect the language first. The input may contain a mix of sentences for translation and translation requests. Do not proceed with the translation, only perform language detection.
        
        <Format>
        ```json
        {
            "source_language": "The language of the detected sentence",
            "target_language": "The language into which the sentence should be translated"
        }
        Json format with the keys: source_language, target_language.
        </Format>
        """

        chat_prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=input),
            ]
        )

        chain = chat_prompt | self.llm_json | fixing_parser

        response = chain.invoke({"input": input})

        required_keys = {"source_language", "target_language"}
        missing_keys = required_keys - response.keys()
        if missing_keys:
            raise ValueError(f"Response is missing required keys: {missing_keys}")

        return response

    def translate(
        self,
        input: str,
        source_language: str = "en",
        target_language: str = "ko",
    ) -> str:

        if input == "":
            return "번역할 문장을 입력해주세요."

        if source_language == "" or target_language == "":
            return "언어 감지에 실패했습니다. 다시 시도해주세요."

        system_prompt = f"""
        You are an expert translator with a deep understanding of {source_language} and {target_language}.
        Translate the following text from {source_language} to {target_language}.
        Ensure the translation is accurate and retains the original meaning.
        Respond with only the translated text, without any additional commentary or explanation.
        """

        chat_prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=input),
            ]
        )

        chain = chat_prompt | self.llm | StrOutputParser()
        translated_text = chain.invoke({"input": input})

        response = f"""
        알겠습니다. 번역 결과는 다음과 같습니다.\n
        {translated_text}
        """

        return response
