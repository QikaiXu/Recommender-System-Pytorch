import numpy as np
import torch
import torch.nn as nn


# 在 cpu 下，比 nn.Embedding 快，但是在 gpu 的序列模型下比后者慢太多了
class CpuEmbedding(nn.Module):

    def __init__(self, num_embeddings, embed_dim):
        super(CpuEmbedding, self).__init__()

        self.weight = nn.Parameter(torch.zeros((num_embeddings, embed_dim)))
        nn.init.xavier_uniform_(self.weight.data)

    def forward(self, x):
        """
        :param x: shape (batch_size, num_fields)
        :return: shape (batch_size, num_fields, embedding_dim)
        """
        return self.weight[x]


class Embedding:

    def __new__(cls, num_embeddings, embed_dim):
        if torch.cuda.is_available():
            embedding = nn.Embedding(num_embeddings, embed_dim)
            nn.init.xavier_uniform_(embedding.weight.data)
            return embedding
        else:
            return CpuEmbedding(num_embeddings, embed_dim)


class FeaturesEmbedding(nn.Module):

    def __init__(self, field_dims, embed_dim):
        super(FeaturesEmbedding, self).__init__()
        self.embedding = Embedding(sum(field_dims), embed_dim)

        # e.g. field_dims = [2, 3, 4, 5], offsets = [0, 2, 5, 9]
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)

    def forward(self, x):
        """
        :param x: shape (batch_size, num_fields)
        :return: shape (batch_size, num_fields, embedding_dim)
        """
        x = x + x.new_tensor(self.offsets)
        return self.embedding(x)


class EmbeddingsInteraction(nn.Module):

    def __init__(self):
        super(EmbeddingsInteraction, self).__init__()

    def forward(self, x):
        """
        :param x: shape (batch_size, num_fields, embedding_dim)
        :return: shape (batch_size, num_fields*(num_fields)//2, embedding_dim)
        """

        num_fields = x.shape[1]
        i1, i2 = [], []
        for i in range(num_fields):
            for j in range(i + 1, num_fields):
                i1.append(i)
                i2.append(j)
        interaction = torch.mul(x[:, i1], x[:, i2])

        return interaction


class MultiLayerPerceptron(nn.Module):

    def __init__(self, layer, batch_norm=True):
        super(MultiLayerPerceptron, self).__init__()
        layers = []
        input_size = layer[0]
        for output_size in layer[1: -1]:
            layers.append(nn.Linear(input_size, output_size))
            if batch_norm:
                layers.append(nn.BatchNorm1d(output_size))
            layers.append(nn.ReLU())
            input_size = output_size
        layers.append(nn.Linear(input_size, layer[-1]))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)
