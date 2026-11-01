import json
import re
from dataclasses import dataclass

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from src.config import Settings
from src.errors import FinSightError, provider_error
from .prompts import SYSTEM_PROMPT
from .vector_store import Evidence

INSUFFICIENT = "The uploaded documents do not provide enough information to answer this question."


class AnswerPayload(BaseModel):
    answer: str = Field(description="Evidence-grounded answer with inline evidence IDs")
    citation_ids: list[str] = Field(description="Supporting IDs, e.g. E1, E2")
    insufficient: bool = Field(description="True when the evidence does not support the answer")


@dataclass(frozen=True)
class Answer:
    question: str
    text: str
    sources: list[dict]
    evidence: list[Evidence]
    insufficient: bool


def chat_client(settings: Settings):
    if not settings.configured:
        raise FinSightError("Configure GEMINI_API_KEY and model names securely before asking questions.")
    return ChatGoogleGenerativeAI(model=settings.chat_model, google_api_key=settings.api_key,
                                 vertexai=False, temperature=0, timeout=45, max_retries=1).with_structured_output(AnswerPayload)


def answer_question(question: str, evidence: list[Evidence], client) -> Answer:
    question = question.strip()
    if not question or len(question) > 2000:
        raise FinSightError("Enter a question between 1 and 2,000 characters.")
    if not evidence:
        return Answer(question, INSUFFICIENT, [], [], True)
    passages = [{"id": f"E{i}", "source": item.source, "text": item.text} for i, item in enumerate(evidence, 1)]
    try:
        response = client.invoke([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=json.dumps({"question": question, "untrusted_evidence": passages}, ensure_ascii=False))])
        payload = response if isinstance(response, AnswerPayload) else AnswerPayload.model_validate(response)
    except Exception as exc:
        raise provider_error(exc) from None
    ids = list(dict.fromkeys(payload.citation_ids))
    valid = {item["id"]: item for item in passages}
    if payload.insufficient or not ids or any(label not in valid for label in ids) or not payload.answer.strip():
        return Answer(question, INSUFFICIENT, [], evidence, True)
    # The model must make its cited IDs visible. Citations prove provenance, not truth.
    inline_ids = set(re.findall(r"\bE\d+\b", payload.answer))
    if inline_ids != set(ids):
        return Answer(question, INSUFFICIENT, [], evidence, True)
    sources = [{"id": label, "source": valid[label]["source"]} for label in ids]
    return Answer(question, payload.answer.strip(), sources, evidence, False)
