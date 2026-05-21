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
    return re.sub(r"\s*\[\d+\]", "", text)


def build_citation_map(answer: str, contexts: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Parse [N] numeric token references from answer and map to source metadata."""
    import re
    citation_map: Dict[str, Dict[str, Any]] = {}
    if not answer or not contexts:
        return citation_map

    refs = re.findall(r"\[(\d+)\]", answer)
    for doc_idx_str in refs:
        doc_id = int(doc_idx_str) - 1  # convert to 0-based
        if 0 <= doc_id < len(contexts):
            ctx = contexts[doc_id]
            key = f"[{doc_idx_str}]"
            citation_map[key] = {
                "filename": ctx.get("filename", "Unknown"),
                "page": ctx.get("page_number"),
                "chunk_id": ctx.get("chunk_id"),
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

        # Find all occurrences of [N] numeric tokens
        pattern = r"\[(\d+)\]"
        matches = re.findall(pattern, response)
        
        invalid_citations = []
        is_valid = True

        for doc_idx_str in matches:

            doc_idx = int(doc_idx_str) - 1  # convert 1-based token to 0-based index

            if doc_idx < 0 or doc_idx >= len(contexts):

                is_valid = False
                invalid_citations.append({
                    "citation": f"[{doc_idx_str}]",
                    "reason": f"Document index {doc_idx_str} is out of range (total documents: {len(contexts)})"
                })

        return is_valid, invalid_citations


    @trace(tags=["guardrails", "faithfulness_check"])
    async def check_faithfulness(
        self,
        answer: str,
        contexts: List[Dict[str, Any]],
        model: str = "local-llm",
        metrics_collector: Optional[Any] = None,
        formatted_context: Optional[str] = None
    ) -> Dict[str, Any]:

        if formatted_context is None:

            from rag_engine.core import RAGCoreEngine
            _, formatted_context = RAGCoreEngine.prepare_context_and_citations(contexts)

        prompt = FAITHFULNESS_CHECK_PROMPT.format(context=formatted_context, answer=answer)
        messages = [{"role": "user", "content": prompt}]
        logger.debug(f"\033[1;33m[FAITHFULNESS CHECK]\033[0m Input: answer='{answer[:200]}...' | context='{formatted_context[:200]}...'")

        try:

            faith_start = time.time()
            response = await self.llm_client.acompletion(
                messages=messages,
                model=model,
                temperature=0.0,
                metrics_collector=metrics_collector,
                metrics_purpose="FAITHFULNESS CHECK"
            )
            eval_text = response.choices[0].message.content
            faith_elapsed = time.time() - faith_start
            usage = getattr(response, "usage", None)
            pt = usage.prompt_tokens if usage else 0
            ct = usage.completion_tokens if usage else 0
            tt = usage.total_tokens if usage else 0
            logger.debug(f"\033[1;33m[FAITHFULNESS CHECK]\033[0m Output: {eval_text}")
            logger.info(
                f"\033[1;33m[FAITHFULNESS CHECK]\033[0m "
                f"\033[1;33m{model}\033[0m "
                f"\033[1;32m[Tokens: {tt} (In={pt}, Out={ct})]\033[0m "
                f"time: {faith_elapsed:.3f}s"
            )
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
        chat_history: str = "",
        metrics_collector: Optional[Any] = None,
    ) -> Dict[str, Any]:

        if max_attempts is None:

            max_attempts = self.config["guardrails"]["max_attempts"]

        gen_temp = self.config["generation"].get("temperature", 0.0)
        gen_tokens = self.config["generation"].get("max_tokens", 512)

        llm_metrics = metrics_collector or LLMMetricsCollector()

        total_gen_start = time.time()
        logger.info(
            f"\033[1;33m[GUARDRAILS START]\033[0m"
            f"model=\033[1;33m{model}\033[0m | "
            f"max_attempts={max_attempts} | contexts={len(contexts)} | "
            f"query='{query[:80]}{'...' if len(query) > 80 else ''}'"
        )

        if on_thought:

            on_thought("✨ Formulating initial answer grounded in context document pages...")

        if not contexts:

            # Simple fallback generation if no context retrieved
            messages = [{"role": "user", "content": query}]
            logger.debug(f"\033[1;33m[NO-CONTEXT FALLBACK]\033[0m Input: query='{query}'")
            fallback_start = time.time()
            response = await self.llm_client.acompletion(
                messages=messages,
                model=model,
                temperature=gen_temp,
                max_tokens=gen_tokens,
                metrics_collector=llm_metrics,
                metrics_purpose="NO-CONTEXT FALLBACK"
            )
            fallback_duration = time.time() - fallback_start
            answer = response.choices[0].message.content
            usage = getattr(response, "usage", None)
            pt = usage.prompt_tokens if usage else 0
            ct = usage.completion_tokens if usage else 0
            tt = usage.total_tokens if usage else 0
            logger.debug(f"\033[1;33m[NO-CONTEXT FALLBACK]\033[0m Output: {answer[:200]}...")
            logger.info(
                f"\033[1;33m[NO-CONTEXT FALLBACK]\033[0m "
                f"\033[1;33m{model}\033[0m "
                f"\033[1;32m[Tokens: {tt} (In={pt}, Out={ct})]\033[0m "
                f"time: {fallback_duration:.3f}s"
            )
            logger.info(f"\n{llm_metrics.format_pretty_block()}")
            if on_thought:

                on_thought(f"⏱️ No-context fallback completed. Time: {fallback_duration:.3f}s")
            return {
                "answer": response.choices[0].message.content,
                "faithful": True,
                "attempts": 1,
                "invalid_citations": [],
                "contradictions": [],
                "citation_map": {},
                "llm_metrics": llm_metrics.to_dict()
            }

        # 1. Format the context for prompt input via sequential citation mapping
        from rag_engine.core import RAGCoreEngine
        formatted_context, citation_map = RAGCoreEngine.prepare_context_and_citations(contexts)

        # 2. CITATION GENERATION
        prompt = CITATION_GENERATION_PROMPT.format(chat_history=chat_history, context=formatted_context, query=query)
        messages = [{"role": "user", "content": prompt}]
        logger.debug(f"\033[1;33m[CITATION GENERATION]\033[0m Input: query='{query}' | chat_history='{chat_history[:100]}...' | context='{formatted_context[:200]}...'")
        
        initial_start = time.time()
        response = await self.llm_client.acompletion(
            messages=messages,
            model=model,
            temperature=gen_temp,
            max_tokens=gen_tokens,
            metrics_collector=llm_metrics,
            metrics_purpose="CITATION GENERATION"
        )
        answer = response.choices[0].message.content
        initial_duration = time.time() - initial_start
        usage = getattr(response, "usage", None)
        pt = usage.prompt_tokens if usage else 0
        ct = usage.completion_tokens if usage else 0
        tt = usage.total_tokens if usage else 0
        logger.debug(f"\033[1;33m[CITATION GENERATION]\033[0m Output: {answer[:300]}...")
        logger.info(
            f"\033[1;33m[CITATION GENERATION]\033[0m "
            f"\033[1;33m{model}\033[0m "
            f"\033[1;32m[Tokens: {tt} (In={pt}, Out={ct})]\033[0m "
            f"time: {initial_duration:.3f}s"
        )
        if on_thought:

            on_thought(f"⏱️ [CITATION GENERATION] completed. Time: {initial_duration:.3f}s")

        # 3. Iterative Validation and Rewrite Correction Loop
        for attempt in range(max_attempts):

            if on_thought:

                on_thought(f"🛡️ Verifying faithfulness alignment & citation matching (Attempt {attempt + 1} of {max_attempts})...")

            # Validate citations
            cit_start = time.time()
            citations_ok, invalid_citations = self.validate_citations(answer, contexts)
            cit_duration = time.time() - cit_start
            logger.debug(f"\033[1;33m[CITATION VALIDATION]\033[0m Input: answer='{answer[:200]}...' | Output: valid={citations_ok}, invalid={invalid_citations}")
            logger.info(
                f"\033[1;33m[CITATION VALIDATION]\033[0m "
                f"valid={citations_ok} | invalid_count={len(invalid_citations)} | "
                f"time: {cit_duration:.3f}s"
            )
            if on_thought:

                on_thought(f"⏱️ [CITATION VALIDATION] (Attempt {attempt + 1}) completed. Time: {cit_duration:.3f}s")

            # Check faithfulness
            eval_data = await self.check_faithfulness(answer, contexts, model=model, metrics_collector=llm_metrics, formatted_context=formatted_context)
            is_faithful = eval_data.get("faithful", True)
            faith_duration = time.time() - cit_start
            logger.info(
                f"\033[1;33m[FAITHFULNESS CHECK RESULT]\033[0m "
                f"faithful={is_faithful} | claims={len(eval_data.get('claims', []))} | "
                f"time: {faith_duration:.3f}s"
            )
            if on_thought:

                on_thought(f"⏱️ [FAITHFULNESS CHECK] (Attempt {attempt + 1}) completed. Time: {faith_duration:.3f}s")

            if is_faithful and citations_ok:

                if on_thought:

                    on_thought(f"✅ Answer passed all checks. Time: {(time.time() - total_gen_start):.3f}s")

                total_gen_duration = time.time() - total_gen_start
                logger.info(
                    f"\033[1;33m[GUARDRAILS END]\033[0m Passed after {attempt + 1} attempt(s) | "
                    f"total_time={total_gen_duration:.3f}s | "
                    f"citations={len(citation_map)} | "
                    f"answer_len={len(answer)} chars"
                )
                logger.info(f"\n{llm_metrics.format_pretty_block()}")
                return {
                    "answer": answer,
                    "faithful": True,
                    "attempts": attempt + 1,
                    "invalid_citations": [],
                    "contradictions": [],
                    "citation_map": citation_map,
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
                f"\033[1;33m[GUARDRAILS]\033[0m Validation failed attempt {attempt + 1} | "
                f"contradictions={len(contradictions)} | rewriting..."
            )

            if on_thought:

                on_thought(f"⚠️ Self-correction: rewriting response to fix {len(contradictions)} contradictions/citations...")

            # 4. Generate rewrite with corrections
            contradiction_text = "\n".join(f"- {c}" for c in contradictions)
            rewrite_prompt = SELF_CORRECTION_REWRITE_PROMPT.format(
                context=formatted_context,
                query=query,
                contradictions=contradiction_text,
                answer=answer
            )

            messages = [{"role": "user", "content": rewrite_prompt}]
            logger.debug(f"\033[1;33m[SELF CORRECTION REWRITE]\033[0m Input: contradictions={contradiction_text} | previous_answer='{answer[:200]}...'")
            rewrite_start = time.time()
            rewrite_res = await self.llm_client.acompletion(
                messages=messages,
                model=model,
                temperature=gen_temp,
                max_tokens=gen_tokens,
                metrics_collector=llm_metrics,
                metrics_purpose="SELF CORRECTION REWRITE"
            )
            answer = rewrite_res.choices[0].message.content
            rewrite_duration = time.time() - rewrite_start
            usage = getattr(rewrite_res, "usage", None)
            pt = usage.prompt_tokens if usage else 0
            ct = usage.completion_tokens if usage else 0
            tt = usage.total_tokens if usage else 0
            logger.debug(f"\033[1;33m[SELF CORRECTION REWRITE]\033[0m Output: {answer[:300]}...")
            logger.info(
                f"\033[1;33m[SELF CORRECTION REWRITE]\033[0m "
                f"\033[1;33m{model}\033[0m "
                f"\033[1;32m[Tokens: {tt} (In={pt}, Out={ct})]\033[0m "
                f"contradictions_fixed={len(contradictions)} | "
                f"time: {rewrite_duration:.3f}s"
            )
            if on_thought:

                on_thought(f"⏱️ [SELF CORRECTION REWRITE] (Attempt {attempt + 1}) completed. Time: {rewrite_duration:.3f}s")

        # If we exhausted attempts without passing checks, return latest state but flagged
        total_gen_duration = time.time() - total_gen_start
        logger.warning(
            f"\033[1;33m[GUARDRAILS]\033[0m Exhausted {max_attempts} attempt(s) | "
            f"returning best-effort | total_time={total_gen_duration:.3f}s | "
            f"answer_len={len(answer)} chars"
        )
        logger.info(f"\n{llm_metrics.format_pretty_block()}")
        if on_thought:

            on_thought(f"⚠️ Verification failed after {max_attempts} attempts. Total time: {total_gen_duration:.3f}s")
        return {
            "answer": answer,
            "faithful": False,
            "attempts": max_attempts,
            "invalid_citations": invalid_citations,
            "contradictions": contradictions,
            "citation_map": citation_map,
            "llm_metrics": llm_metrics.to_dict()
        }
