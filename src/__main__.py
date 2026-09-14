from .tokenizer import Tokenizer


if __name__ == "__main__":
    with open("training_data/pg84.txt", "r") as f:
        training_data = f.read()

    tokenizer = Tokenizer()
    tokenizer.train(training_data, 5000)
