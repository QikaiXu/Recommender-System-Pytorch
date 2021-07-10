from layer.layer import Embedding, FeaturesEmbedding, EmbeddingsInteraction, MultiLayerPerceptron

import torch
import torch.nn as nn


class WideDeep(nn.Module):

    def __init__(self, field_dims, embed_dim=4):
        super(WideDeep, self).__init__()

        self.wide = FeaturesEmbedding(field_dims, 1)

        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.deep = MultiLayerPerceptron([embed_dim * len(field_dims), 128, 64, 32])
        self.fc = nn.Linear(32 + embed_dim * len(field_dims), 1)

    def forward(self, x):
        # x shape: (batch_size, num_fields)
        wide_output = self.wide(x)
        embedding_output = self.embedding(x).reshape(x.shape[0], -1)
        deep_output = self.deep(embedding_output)
        concat = torch.hstack([embedding_output, deep_output])
        output = self.fc(concat)
        output = torch.sigmoid(output)

        return output

