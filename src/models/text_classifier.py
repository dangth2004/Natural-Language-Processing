from typing import List, Dict
from src.core.interfaces import Vectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class TextClassifier:
    def __init__(self, vectorizer: Vectorizer):
        self._vectorizer = vectorizer
        self._model = None

    def fit(self, texts: List[str], labels: List[int]) -> None:
        X = self._vectorizer.fit_transform(texts)
        self._model = LogisticRegression(solver='liblinear', random_state=42).fit(X, labels)

    def predict(self, texts: List[str]) -> List[int]:
        if self._model is None:
            raise RuntimeError("Model is not trained. Please call 'fit()' first.")

        X = self._vectorizer.transform(texts)
        predictions = self._model.predict(X)
        return list(predictions)

    def evaluate(self, y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        metrics = {"accuracy": accuracy, "precision": precision,
                   "recall": recall, "f1_score": f1}

        return metrics
