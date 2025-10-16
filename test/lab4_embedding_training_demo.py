import os
import time
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess


# 1. Stream the data efficiently
def read_corpus(file_path):
    """Yields a list of tokens for each line in the corpus file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield simple_preprocess(line)  # tokenizes + lowercases


# 2. Train the Word2Vec model
def train_word2vec(corpus_path, model_save_path):
    print("Reading training data...")
    sentences = list(read_corpus(corpus_path))
    print(f"Loaded {len(sentences)} sentences.")

    print("Training Word2Vec model...")
    start_time = time.perf_counter()
    model = Word2Vec(
        sentences=sentences,
        vector_size=100,  # embedding dimension
        window=15,  # context window
        min_count=2,  # ignore rare words
        workers=8,  # number of CPU cores
        sg=1  # 1 = skip-gram, 0 = CBOW
    )
    end_time = time.perf_counter()
    print(f"Training time: {end_time - start_time} seconds.")

    print(f"Saving model to {model_save_path}")
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    model.save(model_save_path)
    print("Model saved successfully!")
    return model


# 3. Demonstrate model usage
def demo_model(model):
    print("\n=== Example Queries ===")

    word = "computer"
    if word in model.wv:
        print(f"\nTop 5 words similar to '{word}':")
        for w, sim in model.wv.most_similar(word, topn=5):
            print(f"  {w:15} {sim:.4f}")
    else:
        print(f"'{word}' not found in vocabulary.")

    # Analogy example: king - man + woman ≈ ?
    if all(w in model.wv for w in ["king", "man", "woman"]):
        print("\nAnalogy: king - man + woman ≈ ?")
        result = model.wv.most_similar(positive=["king", "woman"], negative=["man"], topn=1)
        print(f"  Result: {result[0][0]} ({result[0][1]:.4f})")
    else:
        print("\nSome words missing for analogy test.")


# 4. Main execution
def main():
    corpus_path = "/home/dangth2004/Programming/Natural-Language-Processing/data/UD_English-EWT/en_ewt-ud-train.conllu"
    res_path = "/home/dangth2004/Programming/Natural-Language-Processing/results"
    model_path = os.path.join(res_path, "word2vec_ewt.model")

    if not os.path.exists(corpus_path):
        print(f"File not found: {corpus_path}")
        return

    model = train_word2vec(corpus_path, model_path)
    demo_model(model)


if __name__ == "__main__":
    main()
