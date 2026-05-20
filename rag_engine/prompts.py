# Externalized Prompt Templates for RAG Engine

HYDE_GENERATION_PROMPT = """Given the query: '{query}', write a detailed, hypothetical document that answers this query directly. The document should contain relevant facts, terminology, and be optimized for semantic search retrieval.

Write a hypothetical document below."""


CITATION_GENERATION_PROMPT = """System: You are an expert RAG generator. Answer the user query based ONLY on the provided context nodes. Ground every single claim you make with precise inline citations in the format `[Doc-X, p. Y]`, where X is the 0-based document index and Y is the page number. If the document does not specify a page, use `[Doc-X, p. 1]`. Never make claims that are not fully supported by the context.

{chat_history}
Context:
{context}

Query: {query}

Write your answer below with [Doc-X, p. Y] citations."""


FAITHFULNESS_CHECK_PROMPT = """System: You are an independent RAG faithfulness evaluator. Your task is to analyze the generated answer against the retrieved context nodes. Break down the answer into individual claims, check if each claim is fully supported (entailed) by the context, and output a JSON response.
Do not assume or extrapolate. Every claim must be directly supported by the context.

Return ONLY a valid JSON object in this format:
{{
  "faithful": true,
  "claims": [
    {{
      "claim": "Claim text here",
      "supported": true,
      "contradiction_details": ""
    }}
  ]
}}
If any claim is not supported, set "faithful" to false and describe the contradiction/hallucination.

Context:
{context}

Answer:
{answer}

Respond with valid JSON below."""


SELF_CORRECTION_REWRITE_PROMPT = """System: You are an expert RAG self-correction engine. The previous generated answer has failed a faithfulness validation check because it contains claims not supported by the context.
Rewrite the answer so that it is 100% faithful to the context. Remove or correct any claims flagged as unsupported. Do not introduce new unsupported claims. Ground every claim with inline citations in the format `[Doc-X, p. Y]`.

Context:
{context}

Original Query: {query}

Unsupported Claims / Contradiction Details:
{contradictions}

Previous Answer:
{answer}

Write the corrected answer below."""
