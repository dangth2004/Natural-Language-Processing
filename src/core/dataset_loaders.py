def load_raw_text_data(file_path: str) -> str:
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            raw_text = []
            for line in file:
                text = line.strip()
                raw_text.append(text)
            # Join all sentences with a space
            return ' '.join(raw_text)
    except FileNotFoundError:
        raise FileNotFoundError(f"The file {file_path} was not found.")
    except IOError as e:
        raise IOError(f"Error reading file {file_path}: {str(e)}")
