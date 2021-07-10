from layer.layer import Embedding, FeaturesEmbedding, EmbeddingsInteraction, MultiLayerPerceptron

import torch
import torch.nn as nn


class AttentionNet(nn.Module):

    def __init__(self, embed_dim=4, t=4):
        super(AttentionNet, self).__init__()

        self.an = nn.Sequential(
            nn.Linear(embed_dim, t),  # (batch_size, num_crosses, t), num_crosses = num_fields*(num_fields-1)//2
            nn.ReLU(),
            nn.Linear(t, 1, bias=False),  # (batch_size, num_crosses, 1)
            nn.Flatten(),  # (batch_size, num_crosses)
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.an(x)


class AttentionalFactorizationMachine(nn.Module):

    def __init__(self, field_dims, embed_dim=4):
        super(AttentionalFactorizationMachine, self).__init__()

        num_fields = len(field_dims)

        self.w0 = nn.Parameter(torch.zeros((1,)))

        self.embed1 = FeaturesEmbedding(field_dims, 1)
        self.embed2 = FeaturesEmbedding(field_dims, embed_dim)
        self.interact = EmbeddingsInteraction()

        self.attention = AttentionNet(embed_dim)
        self.p = nn.Parameter(torch.zeros(embed_dim, ))
        nn.init.xavier_uniform_(self.p.unsqueeze(0).data)

    def forward(self, x):
        # x size: (batch_size, num_fields)
        # embed(x) size: (batch_size, num_fields, embed_dim)

        embeddings = self.embed2(x)
        interactions = self.interact(embeddings)

        att = self.attention(interactions)
        att_part = interactions.mul(att.unsqueeze(-1)).sum(dim=1).mul(self.p).sum(dim=1, keepdim=True)

        output = self.w0 + self.embed1(x).sum(dim=1) + att_part
        output = torch.sigmoid(output)

        return output


