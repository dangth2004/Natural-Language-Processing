import gensim.downloader as api
import numpy as np

from src.preprocessing.regex_tokenizer import RegexTokenizer


class WordEmbedder:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        try:
            print(f"Loading model '{self.model_name}'... This may take a moment.")
            self.model = api.load(model_name)
            print("Model loaded successfully!")
        except ValueError as e:
            print(f"Error: Model '{self.model_name}' not found. Please choose a valid model.")
            print(f"Available models: {list(api.info()['models'].keys())}")
            raise e

    def get_vector(self, word: str):
        try:
            return self.model[word]
        except KeyError:
            print(f"Warning: Word '{word}' not in vocabulary.")
            return None

    def get_similarity(self, word1: str, word2: str):
        if word1 not in self.model or word2 not in self.model:
            print(f"Warning: One or both words ('{word1}', '{word2}') not in vocabulary.")
            return None
        return self.model.similarity(word1, word2)

    def get_most_similar(self, word: str, top_n: int = 10):
        try:
            return self.model.most_similar(word, topn=top_n)
        except KeyError:
            print(f"Warning: Word '{word}' not in vocabulary.")
            return None

    def embed_document(self, document: str):
        word_vectors = []
        tokenizer = RegexTokenizer()
        tokens = tokenizer.tokenize(document)

        for token in tokens:
            vector = self.get_vector(token)
            if vector is not None:
                word_vectors.append(vector)

        # If the list is empty (no known words), return a zero vector
        if not word_vectors:
            print("⚠️ Warning: Document contains no words in the vocabulary. Returning zero vector.")
            return np.zeros(self.model.vector_size)

        # Compute the element-wise mean of all vectors
        return np.mean(word_vectors, axis=0)
