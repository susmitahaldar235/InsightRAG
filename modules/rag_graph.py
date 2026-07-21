from langgraph.graph import StateGraph, END

from modules.rag_state import RAGState
from modules.rag_nodes import RAGNodes


class RAGGraph:

    def __init__(self, retriever, generator, corrective_rag):

        nodes = RAGNodes(
            retriever,
            generator,
            corrective_rag
        )

        workflow = StateGraph(RAGState)

        # Nodes
        workflow.add_node("retrieve", nodes.retrieve)
        workflow.add_node("grade", nodes.grade)
        workflow.add_node("rewrite", nodes.rewrite)
        workflow.add_node("generate", nodes.generate)

        # Start
        workflow.set_entry_point("retrieve")

        # retrieve -> grade
        workflow.add_edge("retrieve", "grade")

        # Conditional edge
        workflow.add_conditional_edges(
            "grade",
            self.route_after_grade,
            {
                "rewrite": "rewrite",
                "generate": "generate",
            },
        )

        # rewrite -> generate
        workflow.add_edge("rewrite", "generate")

        # generate -> END
        workflow.add_edge("generate", END)

        self.graph = workflow.compile()

    def route_after_grade(self, state: RAGState):

        if state["retry"]:
            return "rewrite"

        return "generate"