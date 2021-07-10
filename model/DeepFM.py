from layer.layer import Embedding, FeaturesEmbedding, EmbeddingsInteraction, MultiLayerPerceptron

import torch
import torch.nn as nn


class DeepFM(nn.Module):

    def __init__(self, field_dims, embed_dim=4):
        super(DeepFM, self).__init__()

        num_fileds = len(field_dims)

        self.embed1 = FeaturesEmbedding(field_dims, 1)
        self.embed2 = FeaturesEmbedding(field_dims, embed_dim)

        self.fm = EmbeddingsInteraction()

        self.deep = MultiLayerPerceptron([embed_dim * num_fileds, 128, 64, 32])
        self.fc = nn.Linear(1 + num_fileds * (num_fileds - 1) // 2 + 32, 1)

    def forward(self, x):
        # x shape: (batch_size, num_fields)
        # embed(x) shape: (batch_size, num_fields, embed_dim)

        embeddings = self.embed2(x)
        embeddings_cross = self.fm(embeddings).sum(dim=-1)
        deep_output = self.deep(embeddings.reshape(x.shape[0], -1))

        stacked = torch.hstack([self.embed1(x).sum(dim=1), embeddings_cross, deep_output])
        output = self.fc(stacked)
        output = torch.sigmoid(output)
        return output

