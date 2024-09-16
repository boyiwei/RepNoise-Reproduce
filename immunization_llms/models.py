import math

import torch
from torch import nn

class classifier(nn.Module):
    def __init__(self, input_dim, labels=2, dtype=torch.bfloat16):
        super(classifier, self).__init__()
        self.up = torch.nn.Linear(input_dim, labels).type(dtype)

        with torch.no_grad():
            nn.init.kaiming_uniform_(self.up.weight, a=math.sqrt(5))
            nn.init.zeros_(self.up.bias)

    def forward(self, x):
        output = self.up(x)
        return output

# This is actually a white-box extension to "Making Harmful Behaviours Unlearnable"
class OptimalNoise(torch.nn.Module):
    def __init__(self, embed_example):
        super().__init__()
        self.noise = torch.nn.Parameter(torch.randn_like(embed_example), requires_grad=True)

    def forward(self):
        return self.noise


class OptimalNoiseEmbedding(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, pad_token_id):
        super().__init__()
        self.embedding = torch.nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_token_id
        )
        self.embedding.weight.data.normal_(mean=0.0, std=1.0)
        self.embedding.weight.requires_grad = True

    def forward(self, input_ids):
        return self.embedding(input_ids)