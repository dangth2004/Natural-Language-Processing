import re
from src.core.interfaces import Tokenizer


class SimpleTokenizer(Tokenizer):
    def tokenize(self, text: str) -> list[str]:
        result = text.lower().split()
        return result
