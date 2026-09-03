from src.bytes_maper import BytesMaper


def main():
    maper = BytesMaper()
    text = "The quick brown fox jumps over 13 lazy dogs 🐶🌟, while saying 'Hello 🌍!' at 3:45 PM."
    ids = maper.encode(text)
    print(ids)
    print(maper.decode(ids))
    print(text == maper.decode(ids))


if __name__ == "__main__":
    main()
