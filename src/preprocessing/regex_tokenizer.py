import re
from src.core.interfaces import Tokenizer


class RegexTokenizer(Tokenizer):
    def tokenize(self, text: str) -> list[str]:
        result = text.lower()
        regex = r"\w+|[^\w\s]"
        result = re.findall(regex, result)

        return result
