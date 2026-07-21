def build_prompt(question, contexts, metadata):
    
    formatted_context = ""

    for i, (doc, meta) in enumerate(zip(contexts, metadata), start=1):

        formatted_context += f"""
Source {i}
Document : {meta['source']}
Chunk ID : {meta['chunk_id']}

Content:
{doc}

------------------------------------
"""

    prompt = f"""
You are an AI assistant.

Answer ONLY from the provided context.

If the answer is not found,
say

"I couldn't find enough information."

At the end of your answer,
add a section

Sources

mention the source document names.

Context

{formatted_context}

Question

{question}

Answer
"""

    return prompt