from dataclasses import dataclass, field
from typing import List


@dataclass
class LLMCallMetrics:
    purpose: str
    elapsed_ms: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    model: str = ""


CYAN = "\033[1;36m"
GREEN = "\033[1;32m"
YELLOW = "\033[1;33m"
WHITE = "\033[1;37m"
RESET = "\033[0m"


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
        model: str = "",
    ):

        self.calls.append(
            LLMCallMetrics(
                purpose=purpose,
                elapsed_ms=elapsed_ms,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                model=model,
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
                    "model": c.model,
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

    def format_pretty_block(self) -> str:

        if not self.calls:
            return ""

        lines = []
        lines.append(
            f"{YELLOW}╔══════════════════ LLM Metrics ══════════════════╗{RESET}"
        )
        for i, c in enumerate(self.calls, 1):
            sec = c.elapsed_ms / 1000
            model_tag = c.model or "?"
            lines.append(
                f"{YELLOW}║{RESET} "
                f" {i:2d}.  {WHITE}{c.purpose:<28}{RESET}"
                f" {GREEN}{model_tag:<12}{RESET}"
                f" {CYAN}{sec:>5.1f}s{RESET}"
                f"  {GREEN}{c.total_tokens:>4}{RESET}  "
                f"(I={c.prompt_tokens}, O={c.completion_tokens})"
            )
        total_sec = self.total_llm_time_ms / 1000
        lines.append(
            f"{YELLOW}║  ───────────────────────────────────────────────{RESET}"
        )
        lines.append(
            f"{YELLOW}║{RESET}  Total: "
            f"{WHITE}{self.total_calls} calls{RESET} · "
            f"{GREEN}{self.total_tokens} tokens{RESET} · "
            f"{CYAN}{total_sec:.1f}s{RESET}"
        )
        lines.append(
            f"{YELLOW}╚══════════════════════════════════════════════════╝{RESET}"
        )
        return "\n".join(lines)
