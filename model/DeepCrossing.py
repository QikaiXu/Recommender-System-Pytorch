from layer.layer import Embedding, FeaturesEmbedding, EmbeddingsInteraction, MultiLayerPerceptron

import torch
import torch.nn as nn


class ResidualUnit(nn.Module):

    def __init__(self, input_size):
        super(ResidualUnit, self).__init__()

        self.fc1 = nn.Linear(input_size, input_size)
        self.fc2 = nn.Linear(input_size, input_size)

    def forward(self, x):
        output = self.fc1(x)
        output = torch.relu(output)
        output = self.fc2(x)
        output = output + x
        return output


class DeepCrossing(nn.Module):

    def __init__(self, field_dims, embed_dim=4, num_res=1):
        super(DeepCrossing, self).__init__()

        input_size = len(field_dims) * embed_dim
        self.res = nn.Sequential(*[ResidualUnit(input_size) for _ in range(num_res)])
        self.embed = FeaturesEmbedding(field_dims, embed_dim)
        self.fc = nn.Linear(input_size, 1)

    def forward(self, x):
        # x shape: (batch_size, num_fields)
        x = self.embed(x)
        x = x.reshape(x.shape[0], -1)
        x = self.res(x)
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x

