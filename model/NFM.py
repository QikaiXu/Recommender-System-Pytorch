from layer.layer import Embedding, FeaturesEmbedding, EmbeddingsInteraction, MultiLayerPerceptron

import torch
import torch.nn as nn


class NeuralFactorizationMachine(nn.Module):

    def __init__(self, field_dims, embed_dim=4):
        super(NeuralFactorizationMachine, self).__init__()

        self.w0 = nn.Parameter(torch.zeros((1,)))

        self.embed1 = FeaturesEmbedding(field_dims, 1)
        self.embed2 = FeaturesEmbedding(field_dims, embed_dim)
        self.interaction = EmbeddingsInteraction()
        self.mlp = MultiLayerPerceptron([embed_dim, 256, 128, 1])

    def forward(self, x):
        # x shape: (batch_size, num_fields)
        # embed(x) shape: (batch_size, num_fields, embed_dim)

        embeddings = self.embed2(x)

        bi_output = self.interaction(embeddings).sum(dim=1)
        f_output = self.mlp(bi_output)

        output = self.w0 + self.embed1(x).sum(dim=1) + f_output
        output = torch.sigmoid(output)

        return output


