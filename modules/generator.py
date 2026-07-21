from modules.llm import GeminiLLM
from modules.prompts import build_prompt


class Generator:

    def __init__(self):

        self.llm = GeminiLLM()

    def generate_answer(self, question, retrieved_docs,metadata):

        prompt = build_prompt(
        question,
        retrieved_docs,
        metadata
        )

        return self.llm.generate(prompt)