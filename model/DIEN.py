from layer.layer import Embedding, FeaturesEmbedding, EmbeddingsInteraction, MultiLayerPerceptron

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence


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


class Attention(nn.Module):

    def __init__(self, embed_dims):
        super(Attention, self).__init__()
        embed_dim1, embed_dim2 = embed_dims
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim1 + embed_dim2 + embed_dim1 * embed_dim2, 36),
            Dice(),
            nn.Linear(36, 1),
        )

    def forward(self, packed: PackedSequence, key):
        # key shape: (batch_size, embed_dim)
        # x shape: (num_x, embed_dim)
        x, batch_sizes, sorted_indices, unsorted_indices = packed
        key = key[sorted_indices]
        idx_list = []
        for batch_size in batch_sizes:
            idx_list.extend(range(batch_size))
        key = key[idx_list]

        # outer product
        i1, i2 = [], []
        for i in range(x.shape[-1]):
            for j in range(key.shape[-1]):
                i1.append(i)
                i2.append(j)
        p = x[:, i1].mul(key[:, i2]).reshape(x.shape[0], -1)

        att = self.mlp(torch.hstack([x, p, key]))
        return att


class AUGRUCell(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(AUGRUCell, self).__init__()

        self.update_gate = nn.Sequential(
            nn.Linear(input_size + hidden_size, 1),
            nn.Sigmoid()
        )
        self.reset_gate = nn.Sequential(
            nn.Linear(input_size + hidden_size, 1),
            nn.Sigmoid()
        )

        self.candidate = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.Tanh()
        )

    def forward(self, x, h, att):
        u = self.update_gate(torch.hstack([x, h]))
        u = att * u
        r = self.reset_gate(torch.hstack([x, h]))
        tilde_h = self.candidate(torch.hstack([x, h * r]))
        h = (1 - u) * h + u * tilde_h
        return h


class AUGRU(nn.Module):

    def __init__(self, input_size, hidden_size, embed_dim=4):
        super(AUGRU, self).__init__()
        self.hidden_size = hidden_size

        self.attention = Attention([hidden_size, embed_dim])
        self.augru_cell = AUGRUCell(input_size, hidden_size)

    def forward(self, packed: PackedSequence, key, h=None):
        x, batch_sizes, sorted_indices, unsorted_indices = packed
        att = self.attention(packed, key)
        device = x.device
        if h == None:
            h = torch.zeros(batch_sizes[0], self.hidden_size, device=device)

        output = torch.zeros(x.shape[0], self.hidden_size)
        output_h = torch.zeros(batch_sizes[0], self.hidden_size, device=device)

        start = 0
        for batch_size in batch_sizes:
            _x = x[start: start + batch_size]
            _att = att[start: start + batch_size]
            _h = h[:batch_size]
            h = self.augru_cell(_x, _h, _att)
            output[start: start + batch_size] = h
            output_h[:batch_size] = h
            start += batch_size

        return PackedSequence(output, batch_sizes, sorted_indices, unsorted_indices), output_h[unsorted_indices]


class DeepInterestEvolutionNetwork(nn.Module):

    def __init__(self, field_dims, embed_dim=4):
        super(DeepInterestEvolutionNetwork, self).__init__()
        hidden_size = embed_dim
        # 商品 embedding 层
        self.embed = Embedding(field_dims[0], embed_dim)

        self.gru = nn.GRU(embed_dim, hidden_size, batch_first=True)
        self.augru = AUGRU(hidden_size, hidden_size, embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim + hidden_size, 200),
            Dice(),
            nn.Linear(200, 80),
            Dice(),
            nn.Linear(80, 1)
        )

    def forward(self, x, neg_sample=None):
        behaviors_ad_embeddings = self.embed(x)  # (batch_size, num_behaviors+1, embed_dim)

        lengths = (x[:, :-1] > 0).sum(dim=1).cpu()
        packed_behaviors = pack_padded_sequence(behaviors_ad_embeddings, lengths, batch_first=True,
                                                enforce_sorted=False)
        packed_gru_output, _ = self.gru(packed_behaviors)
        augru_output, h = self.augru(packed_gru_output, behaviors_ad_embeddings[:, -1])

        #         h = _.view(_.shape[1], -1)
        concated = torch.hstack([h, behaviors_ad_embeddings[:, -1]])
        output = self.mlp(concated)
        output = torch.sigmoid(output)

        if neg_sample is None:
            return output
        else:
            # auxiliary loss part
            gru_output, _ = pad_packed_sequence(packed_gru_output, batch_first=True)
            gru_embedding = gru_output[:, 1:][neg_sample > 0]

            pos_embedding = behaviors_ad_embeddings[:, 1:-1][neg_sample > 0]
            neg_embedding = self.embed(neg_sample)[neg_sample > 0]

            pred_pos = (gru_embedding * pos_embedding).sum(dim=1)
            pred_neg = (gru_embedding * neg_embedding).sum(dim=1)
            auxiliary_output = torch.sigmoid(torch.cat([pred_pos, pred_neg], dim=0)).reshape(2, -1)

            return output, auxiliary_output
