from layer.layer import Embedding, FeaturesEmbedding, EmbeddingsInteraction, MultiLayerPerceptron

import torch
import torch.nn as nn


class NeuralCollaborativeFiltering(nn.Module):

    def __init__(self, field_dims, embed_dim=4):
        super(NeuralCollaborativeFiltering, self).__init__()
        self.embed1 = FeaturesEmbedding(field_dims, embed_dim)
        self.embed2 = FeaturesEmbedding(field_dims, embed_dim)

        self.mlp = MultiLayerPerceptron([len(field_dims) * embed_dim, 128, 64])
        self.fc = nn.Linear(embed_dim + 64, 1)

    def forward(self, x):
        embeddings1 = self.embed1(x)
        gmf_output = embeddings1[:, 0].mul(embeddings1[:, 1]).squeeze(-1)

        embeddings2 = self.embed2(x)
        mlp_input = embeddings2.reshape(x.shape[0], -1)
        mlp_output = self.mlp(mlp_input)

        concated = torch.hstack([gmf_output, mlp_output])
        output = self.fc(concated)
        output = torch.sigmoid(output)
        return output

