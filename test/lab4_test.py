from src.representations.word_embedder import WordEmbedder


def main():
    embedder = WordEmbedder(model_name='glove-wiki-gigaword-50')
    print("Getting vector for the word 'king':")
    king_vector = embedder.get_vector('king')
    if king_vector is not None:
        print(f"Vector Shape: {king_vector.shape}")
        print(f"Vector Preview: {king_vector[:5]}\n")

    print("Getting similarity scores between 'king' and 'queen' and 'man':")
    sim_king_queen = embedder.get_similarity('king', 'queen')
    sim_king_man = embedder.get_similarity('king', 'man')
    print(f"Similarity('king', 'queen'): {sim_king_queen:.4f}")
    print(f"Similarity('king', 'man'):   {sim_king_man:.4f}\n")

    print("Getting the 10 most similar words to 'computer':")
    similar_words = embedder.get_most_similar('computer', top_n=10)
    if similar_words:
        for i, (word, score) in enumerate(similar_words, 1):
            print(f"{i:2d}. {word:<15} | Score: {score:.4f}")

    sentence = "The queen rules the country."
    print(f"\nEmbedding the sentence: '{sentence}'")
    doc_vector = embedder.embed_document(sentence)
    if doc_vector is not None:
        print(f"Resulting Vector Shape: {doc_vector.shape}")
        print(f"Resulting Vector Preview: {doc_vector[:5]}")


if __name__ == "__main__":
    main()
