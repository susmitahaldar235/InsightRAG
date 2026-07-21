from modules.llm import GeminiLLM

class RetrievalGrader:

    def __init__(self):
        self.llm = GeminiLLM()

    def grade(self, question, documents):

        context = "\n\n".join(
        [doc["document"] for doc in documents]
)

        prompt = f"""
You are a retrieval evaluator.

Question:
{question}

Retrieved Context:
{context}

Determine whether the retrieved context contains enough information
to answer the user's question correctly.

Reply with ONLY one word:

YES

or

NO
"""

        response = self.llm.generate(prompt)

        return response.strip().upper()

class QueryRewriter:
    
    def __init__(self):
        self.llm = GeminiLLM()

    def rewrite(self, question):

        prompt = f"""
You are an expert search query optimizer.

Rewrite the following question so that a retrieval system
can find more relevant documents.

Original Question:
{question}

Return ONLY the rewritten question.
"""

        response = self.llm.generate(prompt)

        return response.strip()

class CorrectiveRAG:
    
    def __init__(self):

        self.grader = RetrievalGrader()
        self.rewriter = QueryRewriter()

    def should_retry(self, question, documents):
    
        decision = self.grader.grade(
            question,
            documents
        )

        if "YES" in decision:
            return False, question

        rewritten_question = self.rewriter.rewrite(
            question
        )

        return True, rewritten_question