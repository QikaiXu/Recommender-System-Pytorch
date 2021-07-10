from layer.layer import Embedding, FeaturesEmbedding, EmbeddingsInteraction, MultiLayerPerceptron

import torch
import torch.nn as nn


class LatentFactorModel(nn.Module):

    def __init__(self, field_dims, embed_dim=4):
        super(LatentFactorModel, self).__init__()
        self.embed = FeaturesEmbedding(field_dims, embed_dim)

    def forward(self, x):
        x = self.embed(x)
        x = x[:, 0].mul(x[:, 1]).sum(dim=1, keepdim=True)
        x = torch.sigmoid(x)
        return x

