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


class Classifier(nn.Module):

    def __init__(self, field_dims, k):
        super(Classifier, self).__init__()

        self.embed = FeaturesEmbedding(field_dims, k)

    def forward(self, x):
        # x shape: (batch_size, num_fields)
        # embed(x) shape: (batch_size, num_fields, embed_dim)
        # output shape: (batch_size, k)
        output = self.embed(x).sum(dim=1)

        # output shape: (batch_size, k)
        output = torch.softmax(output, dim=1)
        return output


class MixedLogisticRegression(nn.Module):

    def __init__(self, field_dims, k=5):
        super(MixedLogisticRegression, self).__init__()

        self.clf = Classifier(field_dims, k)
        self.lr_list = nn.ModuleList([LogisticRegression(field_dims) for _ in range(k)])

    def forward(self, x):
        clf_output = self.clf(x)
        lr_output = torch.zeros_like(clf_output)
        for i, lr in enumerate(self.lr_list):
            lr_output[:, i] = lr(x).squeeze(-1)
        output = torch.mul(clf_output, lr_output).sum(dim=1, keepdim=True)
        return output

