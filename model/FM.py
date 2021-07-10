from layer.layer import Embedding, FeaturesEmbedding, EmbeddingsInteraction, MultiLayerPerceptron

import torch
import torch.nn as nn


class FactorizationMachine(nn.Module):

    def __init__(self, field_dims, embed_dim=4):
        super(FactorizationMachine, self).__init__()

        self.embed1 = FeaturesEmbedding(field_dims, 1)
        self.embed2 = FeaturesEmbedding(field_dims, embed_dim)
        self.bias = nn.Parameter(torch.zeros((1,)))

    def forward(self, x):
        # x shape: (batch_size, num_fields)
        # embed(x) shape: (batch_size, num_fields, embed_dim)
        square_sum = self.embed2(x).sum(dim=1).pow(2).sum(dim=1)
        sum_square = self.embed2(x).pow(2).sum(dim=1).sum(dim=1)
        output = self.embed1(x).squeeze(-1).sum(dim=1) + self.bias + (square_sum + sum_square) / 2
        output = torch.sigmoid(output).unsqueeze(-1)
        return output

