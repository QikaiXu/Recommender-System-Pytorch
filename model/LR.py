from layer.layer import Embedding, FeaturesEmbedding, EmbeddingsInteraction, MultiLayerPerceptron

import torch
import torch.nn as nn


class LogisticRegression(nn.Module):

    def __init__(self, field_dims):
        super(LogisticRegression, self).__init__()

        self.bias = nn.Parameter(torch.zeros((1,)))
        self.embed = FeaturesEmbedding(field_dims, 1)

    def forward(self, x):
        # x shape: (batch_size, num_fields)
        output = self.embed(x).sum(dim=1) + self.bias
        output = torch.sigmoid(output)
        return output
