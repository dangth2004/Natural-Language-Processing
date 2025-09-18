from src.preprocessing.regex_tokenizer import RegexTokenizer
from src.preprocessing.simple_tokenizer import SimpleTokenizer


def main():
    text = ["Hello, world! This is a test.",
            "NLP is fascinating... isn't it?",
            "Let's see how it handles 123 numbers and punctuation!"]

    tokenizer = SimpleTokenizer()
    regexer = RegexTokenizer()
    for input_text in text:
        print(f"Input text: {input_text}")
        simple_token = tokenizer.tokenize(input_text)
        regex_token = regexer.tokenize(input_text)
        print(f"Using SimpleTokenizer: {simple_token}")
        print(f"Using RegexTokenizer: {regex_token}\n")


if __name__ == "__main__":
    main()
