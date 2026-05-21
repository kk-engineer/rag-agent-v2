from dataclasses import dataclass, field
from typing import List


@dataclass
class LLMCallMetrics:
    purpose: str
    elapsed_ms: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class LLMMetricsCollector:

    def __init__(self):

        self.calls: List[LLMCallMetrics] = []

    def add_call(
        self,
        purpose: str,
        elapsed_ms: float,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
    ):

        self.calls.append(
            LLMCallMetrics(
                purpose=purpose,
                elapsed_ms=elapsed_ms,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            )
        )

    @property
    def total_calls(self) -> int:

        return len(self.calls)

    @property
    def total_prompt_tokens(self) -> int:

        return sum(c.prompt_tokens for c in self.calls)

    @property
    def total_completion_tokens(self) -> int:

        return sum(c.completion_tokens for c in self.calls)

    @property
    def total_tokens(self) -> int:

        return sum(c.total_tokens for c in self.calls)

    @property
    def total_llm_time_ms(self) -> float:

        return sum(c.elapsed_ms for c in self.calls)

    def to_dict(self) -> dict:

        return {
            "total_calls": self.total_calls,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "total_llm_time_ms": round(self.total_llm_time_ms, 1),
            "per_call_breakdown": [
                {
                    "purpose": c.purpose,
                    "elapsed_ms": round(c.elapsed_ms, 1),
                    "prompt_tokens": c.prompt_tokens,
                    "completion_tokens": c.completion_tokens,
                    "total_tokens": c.total_tokens,
                }
                for c in self.calls
            ],
            "per_call_breakdown_str": self.format_per_call_breakdown(),
        }

    def format_per_call_breakdown(self) -> str:

        parts = []
        for c in self.calls:
            sec = c.elapsed_ms / 1000
            parts.append(
                f"{c.purpose}({sec:.1f}s, {c.total_tokens} tks)"
            )
        return ", ".join(parts)
