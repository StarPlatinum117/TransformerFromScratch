from typing import Optional


class SimpleTokenizer:
    def __init__(self, special_tokens=None):
        if special_tokens is None:
            special_tokens = ["<pad>", "<s>", "</s>"]
        self.special_tokens = special_tokens
        self.word2id = {tok: i for i, tok in enumerate(special_tokens)}
        self.id2word = {i: tok for tok, i in self.word2id.items()}
        self.skip_id = set([self.word2id[tok] for tok in special_tokens])

    def build_vocab(self, texts: list[str]) -> None:
        idx = len(self.word2id)
        for text in texts:
            for word in text.strip().split():
                if word not in self.word2id:
                    self.word2id[word] = idx
                    self.id2word[idx] = word
                    idx += 1

    def encode(self, text: str) -> list[int]:
        ids = [self.word2id[w] for w in text.strip().split()]
        return self.bos_id() + ids + self.eos_id()

    def decode(self, ids: list[int]) -> str:
        words = [self.id2word[id] for id in ids if id not in self.skip_id]
        return " ".join(words)

    def bos_id(self):
        return [self.word2id["<s>"]]

    def eos_id(self):
        return [self.word2id["</s>"]]

    def pad_id(self):
        return [self.word2id["<pad>"]]

    def get_vocab_size(self):
        return len(self.word2id)


def get_identity_dataset(trivial: bool = False) -> tuple[list[tuple[list[int], list[int]]], SimpleTokenizer]:
    # Define tiny corpus.
    samples = [
        "hello world",
        "how are you",
        "why are we still here?",
        "just to suffer?",
        "I will do what I must..."
        "You will try!"
    ]
    tokenizer = SimpleTokenizer()
    tokenizer.build_vocab(samples)

    # Encode pairs (input, target) = (sentence, sentence).
    data = [(tokenizer.encode(s), tokenizer.encode(s)) for s in samples]
    if trivial:
        data = [data[0]]  # to test trivial overfitting case

    return data, tokenizer
