from pydantic import BaseModel, Field
from enum import Enum


class DetectTranslationRequest(BaseModel):
    target_sentence: str = Field(
        description="The sentence that requires translation, detected from the given text"
    )
    source_language: str = Field(description="The detected source language")
    target_language: str = Field(
        description="The detected target language for translation"
    )


class TranslationResult(BaseModel):
    translated_text: str = Field(description="번역된 문장")


class FAQInput(BaseModel):
    question: str = Field(description="질문 내용")


class TaskType(str, Enum):
    TRANSLATE = "Translation"
    SUMMARIZE = "Summary"
    ManualQA = "Inquiry"
    GREETING = "Greeting"
    PROOFREAD = "Proofread"
    TITLERECOMMENDATION = "TitleRecommendation"
    EMAILTEMPLATE = "EmailTemplate"
    ETC = "Others"


class RouterResponse(BaseModel):
    task_type: TaskType = Field(
        ...,
        description="Task type must be one of: Translation, Summary, Inquiry, Greeting, Others",
    )


class SummarizationResult(BaseModel):
    summary: str = Field(description="요약된 문장")


class QAResult(BaseModel):
    answer: str = Field(description="답변")


class RephasingResult(BaseModel):
    rephrased_text: str = Field(description="다시 말한 문장")
