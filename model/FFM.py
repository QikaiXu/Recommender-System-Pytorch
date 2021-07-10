from layer.layer import Embedding, FeaturesEmbedding, EmbeddingsInteraction, MultiLayerPerceptron

import torch
import torch.nn as nn


class FieldAwareFactorizationMachine(nn.Module):

    def __init__(self, field_dims, embed_dim=4):
        super(FieldAwareFactorizationMachine, self).__init__()

        self.field_dims = field_dims

        self.bias = nn.Parameter(torch.zeros((1,)))

        self.embed_linear = FeaturesEmbedding(field_dims, 1)
        self.embed_cross = nn.ModuleList([FeaturesEmbedding(field_dims, embed_dim) for _ in field_dims])

    def forward(self, x):
        # x shape: (batch_size, num_fields)
        # embed(x) shape: (batch_size, num_fields, embed_dim)

        num_fields = len(self.field_dims)

        embeddings = [embed(x) for embed in self.embed_cross]
        embeddings = torch.hstack(embeddings)

        i1, i2 = [], []
        for i in range(num_fields):
            for j in range(i + 1, num_fields):
                i1.append(j * num_fields + i)
                i2.append(i * num_fields + j)

        embedding_cross = torch.mul(embeddings[:, i1], embeddings[:, i2]).sum(dim=2).sum(dim=1, keepdim=True)

        output = self.embed_linear(x).sum(dim=1) + self.bias + embedding_cross
        output = torch.sigmoid(output)
        return output

