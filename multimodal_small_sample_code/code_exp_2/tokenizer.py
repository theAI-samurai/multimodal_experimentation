import torch

class SimpleTokenizer:
    def __init__(self, max_length=32):
        self.max_length = max_length
        self.vocab = {"<PAD>": 0, "<UNK>": 1}
        self.vocab_size = 2

    def fit_on_texts(self, texts):
        idx = 2
        for text in texts:
            for word in text.lower().split():
                if word not in self.vocab:
                    self.vocab[word] = idx
                    idx += 1
        self.vocab_size = len(self.vocab)
        print(f"Vocabulary size: {self.vocab_size}")

    def encode(self, text):
        tokens = [self.vocab.get(w, 1) for w in text.lower().split()]
        if len(tokens) < self.max_length:
            tokens += [0] * (self.max_length - len(tokens))
        else:
            tokens = tokens[:self.max_length]
        return torch.tensor(tokens, dtype=torch.long)

    def encode_batch(self, texts):
        return torch.stack([self.encode(text) for text in texts])

    def get_attention_mask(self, tokens):
        """Return mask where 1 = real token, 0 = PAD"""
        return (tokens != 0).long()