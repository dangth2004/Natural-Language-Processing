from src.preprocessing.regex_tokenizer import RegexTokenizer
from src.preprocessing.simple_tokenizer import SimpleTokenizer
from src.core.dataset_loaders import load_raw_text_data


def main():
    text = ["Hello, world! This is a test.",
            "NLP is fascinating... isn't it?",
            "Let's see how it handles 123 numbers and punctuation!"]

    simple_tokenizer = SimpleTokenizer()
    regex_tokenizer = RegexTokenizer()
    for input_text in text:
        print(f"Input text: {input_text}")
        simple_token = simple_tokenizer.tokenize(input_text)
        regex_token = regex_tokenizer.tokenize(input_text)
        print(f"Using SimpleTokenizer: {simple_token}")
        print(f"Using RegexTokenizer: {regex_token}\n")

    dataset_path = "/home/dangth2004/Programming/Natural-Language-Processing/data/UD_English-EWT/en_ewt-ud-train.txt"
    raw_text = load_raw_text_data(dataset_path)
    # Take a small portion of the text for demonstration
    sample_text = raw_text[:500]  # First 500 characters
    print("\n--- Tokenizing Sample Text from UD_English-EWT ---")
    print(f"Original Sample: {sample_text[:100]}...")
    simple_tokens = simple_tokenizer.tokenize(sample_text)
    print(f"SimpleTokenizer Output (first 20 tokens): {simple_tokens[:20]}")
    regex_tokens = regex_tokenizer.tokenize(sample_text)
    print(f"RegexTokenizer Output (first 20 tokens): {regex_tokens[:20]}")


if __name__ == "__main__":
    main()
