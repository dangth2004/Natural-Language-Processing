from src.preprocessing.regex_tokenizer import RegexTokenizer
from src.representations.count_vectorizer import CountVectorizer


def main():
    regex_tokenizer = RegexTokenizer()
    count_vectorizer = CountVectorizer(tokenizer=regex_tokenizer)

    corpus = [
        "I love NLP.",
        "I love programming.",
        "NLP is a subfield of AI."
    ]

    result = count_vectorizer.fit_transform(corpus)
    print(count_vectorizer.vocabulary_)
    for vector in result:
        print(vector)


if __name__ == "__main__":
    main()
