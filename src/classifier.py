import os
from enum import Enum
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

load_dotenv()

MODEL_NAME = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")


class IntentType(str, Enum):
    EXPLAIN = "Explain Code"
    GENERATE = "Generate Code"
    OFF_TOPIC = "Off Topic"


class IntentSchema(BaseModel):
    intent: IntentType = Field(description=(
    "Explain Code: the user wants existing code explained, analysed or debugged. "
    "Generate Code: the user wants new code written. "
    "Off Topic: anything not about programming or software (general knowledge, "
    "chit-chat, geography, etc.)."
))


class RelevanceSchema(BaseModel):
    is_relevant: bool = Field(description="True if context contains information directly useful to answer the query, False otherwise.")
    reasoning: str = Field(description="Brief explanation of why the context is or isn't relevant.")


class WorkflowClassifier:
    def __init__(self, model_name: str = MODEL_NAME):
        api_key = os.getenv("GROQ_API_KEY")
        self.llm = ChatGroq(
            groq_api_key=api_key,
            model_name=model_name,
            temperature=0.0
        )

    def classify_intent(self, query: str) -> IntentType:
        structured_llm = self.llm.with_structured_output(IntentSchema)
        res = structured_llm.invoke(f"Classify this request: '{query}'")
        return res.intent

    def check_relevance(self, query: str, context: str) -> bool:
        structured_llm = self.llm.with_structured_output(RelevanceSchema)
        prompt = f"Query: {query}\n\nRetrieved Context:\n{context}\n\nIs the context relevant?"
        res = structured_llm.invoke(prompt)
        return res.is_relevant


_classifier_instance = None


def get_classifier() -> WorkflowClassifier:
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = WorkflowClassifier()
    return _classifier_instance


def classify(query: str) -> str:
    try:
        classifier = get_classifier()
        intent = classifier.classify_intent(query)
        return intent.value if hasattr(intent, "value") else str(intent)
    except Exception:
        # Fallback heuristic
        q = query.lower()
        if any(w in q for w in ("explain", "walk through", "what does", "how does", "debug", "review", "why")):
            return IntentType.EXPLAIN.value
        return IntentType.GENERATE.value


def route(intent: str) -> str:
    intent_str = str(intent).lower()
    if "explain" in intent_str:
        return "explain_pipeline"
    return "generate_pipeline"