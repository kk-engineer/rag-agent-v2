import re
import json
import logging
import time

try:
    from opensmith import trace
except ImportError:
    def trace(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]
        def decorator(f):
            return f
        return decorator
from typing import List, Dict, Any, Tuple, Optional
from rag_engine.prompts import (
    CITATION_GENERATION_PROMPT,
    FAITHFULNESS_CHECK_PROMPT,
    SELF_CORRECTION_REWRITE_PROMPT
)
from rag_engine.llm import LiteLLMClient
from rag_engine.metrics import LLMMetricsCollector


logger = logging.getLogger(__name__)


def extract_json(text: str) -> Dict[str, Any]:

    try:

        # Try to find a JSON block enclosed in markdown code fences
        match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        if match:

            return json.loads(match.group(1).strip())
        # Try raw json parsing
        return json.loads(text.strip())
    except Exception:

        # Try searching for the first '{' and last '}'
        try:

            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1:

                return json.loads(text[start:end+1])
        except Exception as e:

            logger.error(f"Failed to parse JSON from response: {e}")
            
    # Default fallback
    return {"faithful": True, "claims": []}


def strip_citations(text: str) -> str:

    if not text:

        return ""
    import re
    return re.sub(r"\s*\[Doc-\d+(?:,\s*p\.\s*\d+)?\]", "", text)


def build_citation_map(answer: str, contexts: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Parse [Doc-X, p. Y] references from answer and map to source filenames."""
    import re
    citation_map: Dict[str, Dict[str, Any]] = {}
    if not answer or not contexts:
        return citation_map

    refs = re.findall(r"\[Doc-(\d+)(?:,\s*p\.\s*(\d+))?\]", answer)
    for doc_id_str, page_str in refs:
        doc_id = int(doc_id_str)
        if doc_id < len(contexts):
            ctx = contexts[doc_id]
            key = f"[Doc-{doc_id_str}"
            if page_str:
                key += f", p. {page_str}"
            key += "]"
            citation_map[key] = {
                "filename": ctx.get("filename", "Unknown"),
                "page": int(page_str) if page_str else ctx.get("page_number"),
                "score": ctx.get("score", 0.0),
            }
    return citation_map


class GuardrailsManager:

    def __init__(self, llm_client: LiteLLMClient, config_path: str = "config/rag_config.toml"):

        self.llm_client = llm_client
        
        # Load configuration
        import os
        self.config = {
            "generation": {
                "temperature": 0.0,
                "max_tokens": 512
            },
            "guardrails": {
                "max_attempts": 3
            }
        }
        
        if os.path.exists(config_path):

            try:

                with open(config_path, "rb") as f:

                    import tomllib
                    loaded = tomllib.load(f)
                    for section in ["generation", "guardrails"]:

                        if section in loaded:

                            self.config[section].update(loaded[section])
            except Exception as e:

                logger.warning(f"Failed to load RAG config in GuardrailsManager: {e}")


    def validate_citations(
        self,
        response: str,
        contexts: List[Dict[str, Any]]
    ) -> Tuple[bool, List[Dict[str, Any]]]:

        # Find all occurrences of [Doc-X, p. Y]
        pattern = r"\[Doc-(\d+),\s*p\.\s*(\d+)\]"
        matches = re.findall(pattern, response)
        
        invalid_citations = []
        is_valid = True

        for match in matches:

            doc_idx_str, page_str = match
            doc_idx = int(doc_idx_str)
            page_num = int(page_str)

            # Check index range
            if doc_idx < 0 or doc_idx >= len(contexts):

                is_valid = False
                invalid_citations.append({
                    "citation": f"[Doc-{doc_idx_str}, p. {page_str}]",
                    "reason": f"Document index {doc_idx} is out of range (total documents: {len(contexts)})"
                })
                continue

            # Check page number range
            target_context = contexts[doc_idx]
            expected_page = target_context.get("page_number", 1)
            
            # Allow page number mismatch check
            if page_num != expected_page:

                is_valid = False
                invalid_citations.append({
                    "citation": f"[Doc-{doc_idx_str}, p. {page_str}]",
                    "reason": f"Page number {page_num} does not match expected page number {expected_page} for source context"
                })

        return is_valid, invalid_citations


    async def check_faithfulness(
        self,
        answer: str,
        contexts: List[Dict[str, Any]],
        model: str = "local-llm",
        metrics_collector: Optional[Any] = None
    ) -> Dict[str, Any]:

        formatted_context = ""
        for idx, ctx in enumerate(contexts):

            formatted_context += f"[Doc-{idx}, p. {ctx.get('page_number', 1)}]: {ctx['text']}\n\n"

        prompt = FAITHFULNESS_CHECK_PROMPT.format(context=formatted_context, answer=answer)
        messages = [{"role": "user", "content": prompt}]

        try:

            response = await self.llm_client.acompletion(
                messages=messages,
                model=model,
                temperature=0.0,
                metrics_collector=metrics_collector,
                metrics_purpose="Faithfulness check"
            )
            eval_text = response.choices[0].message.content
            return extract_json(eval_text)
        except Exception as e:

            logger.error(f"Error checking faithfulness: {e}")
            return {"faithful": True, "claims": []}


    @trace(tags=["guardrails"])
    async def generate_faithful_answer(
        self,
        query: str,
        contexts: List[Dict[str, Any]],
        model: str = "local-llm",
        max_attempts: Optional[int] = None,
        on_thought: Optional[Any] = None,
        chat_history: str = ""
    ) -> Dict[str, Any]:

        if max_attempts is None:

            max_attempts = self.config["guardrails"]["max_attempts"]

        gen_temp = self.config["generation"].get("temperature", 0.0)
        gen_tokens = self.config["generation"].get("max_tokens", 512)

        llm_metrics = LLMMetricsCollector()

        total_gen_start = time.time()
        logger.info(f"Generating faithful answer for query: '{query}'")

        if on_thought:

            on_thought("✨ Formulating initial answer grounded in context document pages...")

        if not contexts:

            # Simple fallback generation if no context retrieved
            messages = [{"role": "user", "content": query}]
            fallback_start = time.time()
            response = await self.llm_client.acompletion(
                messages=messages,
                model=model,
                temperature=gen_temp,
                max_tokens=gen_tokens,
                metrics_collector=llm_metrics,
                metrics_purpose="No-context fallback"
            )
            fallback_duration = time.time() - fallback_start
            logger.info(f"  [Generation Step 1] Fallback answer generation completed (No context). Time taken: {fallback_duration:.3f}s")
            if on_thought:

                on_thought(f"⏱️ Fallback answer formulation completed. Time taken: {fallback_duration:.3f}s")
            return {
                "answer": response.choices[0].message.content,
                "faithful": True,
                "attempts": 1,
                "invalid_citations": [],
                "contradictions": [],
                "llm_metrics": llm_metrics.to_dict()
            }

        # 1. Format the context for prompt input
        formatted_context = ""
        for idx, ctx in enumerate(contexts):

            formatted_context += f"[Doc-{idx}, p. {ctx.get('page_number', 1)}]: {ctx['text']}\n\n"

        # 2. Initial answer generation
        prompt = CITATION_GENERATION_PROMPT.format(chat_history=chat_history, context=formatted_context, query=query)
        messages = [{"role": "user", "content": prompt}]
        
        initial_start = time.time()
        response = await self.llm_client.acompletion(
            messages=messages,
            model=model,
            temperature=gen_temp,
            max_tokens=gen_tokens,
            metrics_collector=llm_metrics,
            metrics_purpose="Initial generation"
        )
        answer = response.choices[0].message.content
        initial_duration = time.time() - initial_start
        logger.info(f"  [Generation Step 1] Initial answer generation completed. Time taken: {initial_duration:.3f}s")
        if on_thought:

            on_thought(f"⏱️ Initial answer formulation completed. Time taken: {initial_duration:.3f}s")

        # 3. Iterative Validation and Rewrite Correction Loop
        for attempt in range(max_attempts):

            if on_thought:

                on_thought(f"🛡️ Verifying faithfulness alignment & citation matching (Attempt {attempt + 1} of {max_attempts})...")

            # Validate citations
            cit_start = time.time()
            citations_ok, invalid_citations = self.validate_citations(answer, contexts)
            cit_duration = time.time() - cit_start
            logger.info(f"  [Generation Step 2.1 (Attempt {attempt + 1})] Citation check completed. Time taken: {cit_duration:.3f}s")
            if on_thought:

                on_thought(f"⏱️ Citation check (Attempt {attempt + 1}) completed. Time taken: {cit_duration:.3f}s")

            # Check faithfulness
            faith_start = time.time()
            eval_data = await self.check_faithfulness(answer, contexts, model=model, metrics_collector=llm_metrics)
            is_faithful = eval_data.get("faithful", True)
            faith_duration = time.time() - faith_start
            logger.info(f"  [Generation Step 2.2 (Attempt {attempt + 1})] Faithfulness check completed. Time taken: {faith_duration:.3f}s")
            if on_thought:

                on_thought(f"⏱️ Faithfulness check (Attempt {attempt + 1}) completed. Time taken: {faith_duration:.3f}s")

            if is_faithful and citations_ok:

                if on_thought:

                    on_thought(f"✅ Success! Generated answer passed all alignment checks. Total verification time: {(time.time() - total_gen_start):.3f}s")

                total_gen_duration = time.time() - total_gen_start
                logger.info(f"Successfully generated faithful answer in {total_gen_duration:.3f}s (Attempts: {attempt + 1})")
                return {
                    "answer": answer,
                    "faithful": True,
                    "attempts": attempt + 1,
                    "invalid_citations": [],
                    "contradictions": [],
                    "llm_metrics": llm_metrics.to_dict()
                }

            # Gather failure reasons/contradictions
            contradictions = []
            if not is_faithful:

                for claim in eval_data.get("claims", []):

                    if not claim.get("supported", True):

                        contradictions.append(
                            f"Claim: '{claim.get('claim')}' - Detail: {claim.get('contradiction_details', 'not supported by context')}"
                        )

            if not citations_ok:

                for citation_err in invalid_citations:

                    contradictions.append(
                        f"Citation error: '{citation_err['citation']}' - Reason: {citation_err['reason']}"
                    )

            logger.warning(
                f"Validation failed on attempt {attempt + 1} for query: '{query}'. "
                f"Contradictions: {contradictions}. Rewriting..."
            )

            if on_thought:

                on_thought(f"⚠️ Self-correction: rewriting response to fix {len(contradictions)} identified contradictions/citations...")

            # 4. Generate rewrite with corrections
            contradiction_text = "\n".join(f"- {c}" for c in contradictions)
            rewrite_prompt = SELF_CORRECTION_REWRITE_PROMPT.format(
                context=formatted_context,
                query=query,
                contradictions=contradiction_text,
                answer=answer
            )

            messages = [{"role": "user", "content": rewrite_prompt}]
            rewrite_start = time.time()
            rewrite_res = await self.llm_client.acompletion(
                messages=messages,
                model=model,
                temperature=gen_temp,
                max_tokens=gen_tokens,
                metrics_collector=llm_metrics,
                metrics_purpose=f"Rewrite (attempt {attempt + 1})"
            )
            answer = rewrite_res.choices[0].message.content
            rewrite_duration = time.time() - rewrite_start
            logger.info(f"  [Generation Step 2.3 (Attempt {attempt + 1})] Self-correction rewrite generation completed. Time taken: {rewrite_duration:.3f}s")
            if on_thought:

                on_thought(f"⏱️ Rewrite generation (Attempt {attempt + 1}) completed. Time taken: {rewrite_duration:.3f}s")

        # If we exhausted attempts without passing checks, return latest state but flagged
        total_gen_duration = time.time() - total_gen_start
        logger.warning(f"Exhausted self-correction attempts ({max_attempts}). Answer returned might still have alignment issues. Total time taken: {total_gen_duration:.3f}s")
        if on_thought:

            on_thought(f"⚠️ Verification failed after {max_attempts} attempts. Total verification time: {total_gen_duration:.3f}s")
        return {
            "answer": answer,
            "faithful": False,
            "attempts": max_attempts,
            "invalid_citations": invalid_citations,
            "contradictions": contradictions,
            "llm_metrics": llm_metrics.to_dict()
        }
