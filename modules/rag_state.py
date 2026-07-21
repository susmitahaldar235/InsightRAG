from typing import TypedDict, List, Dict


class RAGState(TypedDict):
    question: str
    retrieved_docs: List[Dict]
    rewritten_question: str
    retry: bool
    answer: str