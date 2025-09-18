from enum import unique

from src.core.interfaces import Vectorizer, Tokenizer


class CountVectorizer(Vectorizer):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.vocabulary_ = {}

    def fit(self, corpus: list[str]):
        unique_tokens = set()
        for document in corpus:
            tokens = self.tokenizer.tokenize(document)
            unique_tokens.update(tokens)
        self.vocabulary_ = {token: idx for idx, token in enumerate(sorted(unique_tokens))}

    def transform(self, documents: list[str]) -> list[list[int]]:
        result = []
        for document in documents:
            vector = [0] * len(self.vocabulary_)
            tokens = self.tokenizer.tokenize(document)
            for token in tokens:
                if token in self.vocabulary_:
                    vector[self.vocabulary_[token]] += 1
            result.append(vector)
        return result

    def fit_transform(self, corpus: list[str]):
        self.fit(corpus)
        return self.transform(corpus)
