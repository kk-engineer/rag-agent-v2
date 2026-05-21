# Externalized Prompt Templates for RAG Engine

HYDE_GENERATION_PROMPT = """Given the query: '{query}', write a detailed, hypothetical document that answers this query directly. The document should contain relevant facts, terminology, and be optimized for semantic search retrieval.

Write a hypothetical document below."""


CITATION_GENERATION_PROMPT = """System: You are an expert assistant analyzing uploaded source material.
Answer the user query using ONLY the provided context blocks delimited by "--- Document [N] ---".
You MUST cite your sources inline using their respective bracketed numbers, exactly like this: [1] or [2].
If multiple sources support a claim, combine them like this: [1][3].
Do not output any raw database hashes, file names, or trailing bibliographies. Keep citations strictly inline as numeric tokens.

{chat_history}
Context:
{context}

Query: {query}

Write your answer below with numeric citation tokens."""


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
Rewrite the answer so that it is 100% faithful to the context. Remove or correct any claims flagged as unsupported. Do not introduce new unsupported claims. Ground every claim with inline numeric citation tokens like [1] or [2].

Context:
{context}

Original Query: {query}

Unsupported Claims / Contradiction Details:
{contradictions}

Previous Answer:
{answer}

Write the corrected answer below."""


QUERY_ROUTER_PROMPT = """You are an advanced query routing engine for a multi-purpose RAG (Retrieval-Augmented Generation) system. 
Your sole task is to analyze the user's input and determine whether answering it requires pulling context from the uploaded document library 
("RAG_RETRIEVAL") or if it is a basic conversational interaction that can be handled directly by a general LLM ("DIRECT_LLM").

Analyze the input based on these strict definitions:

1. RAG_RETRIEVAL:
- Any substantive question, conceptual inquiry, definition request, or deep analysis.
- Requests for summaries, comparisons, or explanations of specific ideas, theories, historical facts, or technical processes.
- The user is asking *about* a topic, expecting information that would be found within an uploaded book, document, paper, or file.

2. DIRECT_LLM:
- Simple single-word or short greetings, diagnostic test inputs, or casual conversational filler (e.g., "Hi", "Hello", "test", "Thanks!").
- Meta-questions about the AI agent's own identity, functions, or operational state (e.g., "who are you", "what can you do?").
- Abstract, low-context pleasantries or personal status statements (e.g., "who am I", "I am tired").

Output Strategy:
You must respond with a strictly valid JSON object and absolutely nothing else. Do not wrap it in markdown backticks, do not add introductory text, and do not include conversational sign-offs.

Expected JSON Schema:
{{
  "reasoning": "A brief, one-sentence explanation of why the route was chosen.",
  "route": "RAG_RETRIEVAL" or "DIRECT_LLM"
}}

Few-Shot Examples:
---
Input: "Hi"
Output: {{"reasoning": "Simple, short diagnostic greeting token.", "route": "DIRECT_LLM"}}

Input: "Hello"
Output: {{"reasoning": "Standard introductory greeting.", "route": "DIRECT_LLM"}}

Input: "who are you"
Output: {{"reasoning": "Identity meta-question about the AI assistant itself.", "route": "DIRECT_LLM"}}

Input: "who am I"
Output: {{"reasoning": "Conversational test or generic statement regarding the user's identity.", "route": "DIRECT_LLM"}}

Input: "Explain the core concept of the mind-body dualism mentioned in the texts."
Output: {{"reasoning": "Substantive philosophical conceptual question requiring deep thematic analysis from the library documents.", "route": "RAG_RETRIEVAL"}}

Input: "What are the primary arguments presented in chapter 2?"
Output: {{"reasoning": "Explicit request for structural content tracking directly inside the reference material.", "route": "RAG_RETRIEVAL"}}

Input: "Thanks for the help!"
Output: {{"reasoning": "Polite closing remark with no informational lookup intent.", "route": "DIRECT_LLM"}}
---

User Input to Evaluate:
"{user_query}"
"""


DIRECT_LLM_SYSTEM_PROMPT = (
    "You are a helpful AI assistant. "
    "Answer the user's input directly and conversationally. "
    "Do not reference any documents."
)
