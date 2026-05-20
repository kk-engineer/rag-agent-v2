import logging
from typing import List, Dict


logger = logging.getLogger(__name__)


class ConversationMemory:

    def __init__(self, window_size: int = 5):
        self.window_size = max(0, window_size)
        self._messages: List[Dict[str, str]] = []

    def add_turn(self, query: str, answer: str) -> None:
        if not self.window_size:
            return
        self._messages.append({"role": "user", "content": query})
        self._messages.append({"role": "assistant", "content": answer})
        max_messages = self.window_size * 2
        if len(self._messages) > max_messages:
            self._messages = self._messages[-max_messages:]

    def format_history(self) -> str:
        if not self.window_size or not self._messages:
            return ""
        lines = ["Previous conversation:"]
        for msg in self._messages:
            role = "User" if msg["role"] == "user" else "Assistant"
            lines.append(f"{role}: {msg['content']}")
        return "\n".join(lines)

    @property
    def turn_count(self) -> int:
        return len(self._messages) // 2

    def clear(self) -> None:
        self._messages.clear()
