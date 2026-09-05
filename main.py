from src.tokenizer import Tokenizer


def main():
    tokenizer = Tokenizer()

    with open("training_data.txt", "r") as f:
        training_data = f.read()
    tokenizer.train(training_data, 50000)

    # tokenizer.load("tokenizer.json")

    with open("testing_data.txt", "r") as f:
        testing_data = f.read()

    raw_bytes_data = tokenizer.vocab.encode(testing_data)
    data_tokens = tokenizer.encode(testing_data)
    ratio = len(raw_bytes_data) / len(data_tokens)
    print(ratio)


if __name__ == "__main__":
    main()
