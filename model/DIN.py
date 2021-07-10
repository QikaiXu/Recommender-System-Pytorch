from layer.layer import Embedding, FeaturesEmbedding, EmbeddingsInteraction, MultiLayerPerceptron

import torch
import torch.nn as nn


class BaseModel(nn.Module):

    def __init__(self, field_dims, embed_dim=4):
        # 这里只用用户的行为序列作为特征，没用户数据
        super(BaseModel, self).__init__()

        # 商品 embedding 层
        self.embed = Embedding(field_dims[0], embed_dim)
        self.mlp = MultiLayerPerceptron([embed_dim * 2, 200, 80, 1])

    def forward(self, x):
        user_behaviors = x[:, :-1]
        mask = (user_behaviors > 0).float().unsqueeze(-1)
        avg = mask.mean(dim=1, keepdim=True)
        weight = mask.mul(avg)
        user_behaviors_embedding = self.embed(user_behaviors).mul(weight).sum(dim=1)
        ad_embedding = self.embed(x[:, -1])

        concated = torch.hstack([user_behaviors_embedding, ad_embedding])
        output = self.mlp(concated)
        output = torch.sigmoid(output)
        return output


class Dice(nn.Module):

    def __init__(self):
        super(Dice, self).__init__()
        self.alpha = nn.Parameter(torch.zeros((1,)))

    def forward(self, x):
        avg = x.mean(dim=0)
        std = x.std(dim=0)
        norm_x = (x - avg) / std
        p = torch.sigmoid(norm_x)

        return x.mul(p) + self.alpha * x.mul(1 - p)


class ActivationUnit(nn.Module):

    def __init__(self, embed_dim=4):
        super(ActivationUnit, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * (embed_dim + 2), 36),
            Dice(),
            nn.Linear(36, 1),
        )

    def forward(self, x):
        behaviors = x[:, :-1]
        num_behaviors = behaviors.shape[1]

        ads = x[:, [-1] * num_behaviors]

        # outer product
        embed_dim = x.shape[-1]
        i1, i2 = [], []
        for i in range(embed_dim):
            for j in range(embed_dim):
                i1.append(i)
                i2.append(j)
        p = behaviors[:, :, i1].mul(ads[:, :, i2]).reshape(behaviors.shape[0], behaviors.shape[1], -1)

        att = self.mlp(torch.cat([behaviors, p, ads], dim=2))
        return att


class DeepInterestNetwork(nn.Module):

    def __init__(self, field_dims, embed_dim=4):
        super(DeepInterestNetwork, self).__init__()
        # 商品 embedding 层
        self.embed = Embedding(field_dims[0], embed_dim)
        self.attention = ActivationUnit(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, 200),
            Dice(),
            nn.Linear(200, 80),
            Dice(),
            nn.Linear(80, 1)
        )

    def forward(self, x):
        mask = (x > 0).float().unsqueeze(-1)  # (batch_size, num_behaviors+1, 1)
        behaviors_ad_embeddings = self.embed(x).mul(mask)  # (batch_size, num_behaviors+1, embed_dim)
        att = self.attention(behaviors_ad_embeddings)  # (batch_size, num_behaviors, 1)

        weighted_behaviors = behaviors_ad_embeddings[:, :-1].mul(mask[:, :-1]).mul(
            att)  # (batch_size, num_behaviors, embed_dim)
        user_interest = weighted_behaviors.sum(dim=1)  # (batch_size, embed_dim)

        concated = torch.hstack([user_interest, behaviors_ad_embeddings[:, -1]])
        output = self.mlp(concated)
        output = torch.sigmoid(output)
        return output

