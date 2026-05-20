import re
import warnings
import logging

# Suppress Hugging Face/Transformers warning messages and path lookup alerts
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*Accessing.*__path__.*")
warnings.filterwarnings("ignore", message=".*zoedepth.*")

# Silence transformers module logging
logging.getLogger("transformers").setLevel(logging.ERROR)


class ZoedepthWarningFilter(logging.Filter):

    def filter(self, record):

        try:

            msg = record.getMessage()
            if "zoedepth" in msg or "__path__" in msg:

                return False
        except Exception:

            pass
        return True


zoedepth_filter = ZoedepthWarningFilter()
logging.getLogger("transformers").addFilter(zoedepth_filter)
logging.getLogger("py.warnings").addFilter(zoedepth_filter)
logging.getLogger().addFilter(zoedepth_filter)

import sys


class ColoredFormatter(logging.Formatter):

    _level_colors = {
        "DEBUG": "\033[2;37m",
        "INFO": "\033[32m",
        "WARNING": "\033[1;33m",
        "ERROR": "\033[1;31m",
        "CRITICAL": "\033[1;37;41m",
    }

    _module_colors = {
        "rag_engine.core": "\033[36m",
        "rag_engine.llm": "\033[34m",
        "rag_engine.guardrails": "\033[35m",
        "rag_engine.evaluation": "\033[33m",
        "rag_engine.cli": "\033[32m",
        "rag_engine.ui": "\033[37m",
        "rag_engine.utils": "\033[2;37m",
    }

    _msg_colors = {
        "DEBUG": "\033[37m",
        "INFO": "\033[0m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[1;31m",
    }

    _reset = "\033[0m"
    _timing_highlight = "\033[1;36m"

    _timing_re = re.compile(r"(\d+\.\d+s)")

    def _get_module_color(self, name):
        for prefix, color in self._module_colors.items():
            if name == prefix or name.startswith(prefix + "."):
                return color
        return "\033[37m"

    def _highlight_timing(self, message: str) -> str:
        return self._timing_re.sub(
            rf"{self._timing_highlight}\1{self._reset}", message
        )

    def format(self, record):
        timestamp = self.formatTime(record, self.datefmt)
        levelname = record.levelname
        name = record.name
        message = self._highlight_timing(record.getMessage())

        level_color = self._level_colors.get(levelname, "")
        module_color = self._get_module_color(name)
        msg_color = self._msg_colors.get(levelname, "")

        s = (f"\033[2m[{timestamp}]\033[0m"
             f" {level_color}{levelname}{self._reset}"
             f" {module_color}[{name}]{self._reset}"
             f" {msg_color}{message}{self._reset}")

        if record.exc_info and not record.exc_text:
            record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            if s[-1:] != "\n":
                s += "\n"
            s += record.exc_text

        return s


engine_logger = logging.getLogger("rag_engine")
engine_logger.setLevel(logging.INFO)
if not engine_logger.handlers:

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(ColoredFormatter(datefmt="%H:%M:%S"))
    engine_logger.addHandler(handler)
    engine_logger.propagate = False
engine_logger.addFilter(zoedepth_filter)



from rag_engine.llm import LiteLLMClient
from rag_engine.core import RAGCoreEngine, Document
from rag_engine.guardrails import GuardrailsManager
from rag_engine.utils.logger import QueryLogger
from rag_engine.memory import ConversationMemory


__all__ = [
    "LiteLLMClient",
    "RAGCoreEngine",
    "Document",
    "GuardrailsManager",
    "QueryLogger",
    "ConversationMemory",
]
