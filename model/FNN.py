from layer.layer import Embedding, FeaturesEmbedding, EmbeddingsInteraction, MultiLayerPerceptron

import torch
import torch.nn as nn


class FactorizationMachineSupportedNeuralNetwork(nn.Module):

    def __init__(self, field_dims, embed_dim=4):
        super(FactorizationMachineSupportedNeuralNetwork, self).__init__()

        # w1, w2, ..., wn
        self.embed1 = FeaturesEmbedding(field_dims, 1)

        # v1, v2, ..., vn
        self.embed2 = FeaturesEmbedding(field_dims, embed_dim)

        self.mlp = MultiLayerPerceptron([(embed_dim + 1) * len(field_dims), 128, 64, 32, 1])

    def forward(self, x):
        # x shape: (batch_size, num_fields)

        w = self.embed1(x).squeeze(-1)
        v = self.embed2(x).reshape(x.shape[0], -1)
        stacked = torch.hstack([w, v])

        output = self.mlp(stacked)
        output = torch.sigmoid(output)
        return output

