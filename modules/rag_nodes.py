from modules.rag_state import RAGState

class RAGNodes:
    
    def __init__(self, retriever, generator, corrective_rag):
        self.retriever = retriever
        self.generator = generator
        self.corrective_rag = corrective_rag
    
    def retrieve(self, state: RAGState):
    
        print("========== RETRIEVE ==========")

        docs = self.retriever.retrieve(state["question"])

        state["retrieved_docs"] = docs

        return state
    
    def grade(self, state: RAGState):
    
        print("========== GRADE ==========")

        retry, new_question = self.corrective_rag.should_retry(
            state["question"],
            state["retrieved_docs"]
        )

        state["retry"] = retry
        state["rewritten_question"] = new_question

        return state
    
    def rewrite(self, state: RAGState):
    
        print("========== REWRITE ==========")

        docs = self.retriever.retrieve(
            state["rewritten_question"]
        )

        state["retrieved_docs"] = docs

        return state
    
    def generate(self, state: RAGState):
    
        print("========== GENERATE ==========")

        docs = [
            item["document"]
            for item in state["retrieved_docs"]
        ]

        metadata = [
            item["metadata"]
            for item in state["retrieved_docs"]
        ]

        answer = self.generator.generate_answer(
            state["question"],
            docs,
            metadata
        )

        state["answer"] = answer

        return state