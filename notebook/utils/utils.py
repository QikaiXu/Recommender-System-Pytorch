from copy import deepcopy
from tqdm import tqdm
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, KBinsDiscretizer
from sklearn.model_selection import train_test_split
from sklearn import metrics

import torch


class Dataset:

    def __init__(self):
        self.device = torch.device('cpu')

    def to(self, device):
        self.device = device
        return self

    def train_valid_test_split(self, train_size=0.8, valid_size=0.1, test_size=0.1):
        field_dims = (self.data.max(axis=0).astype(int) + 1).tolist()[:-1]

        train, valid_test = train_test_split(self.data, train_size=train_size, random_state=2021)

        valid_size = valid_size / (test_size + valid_size)
        valid, test = train_test_split(valid_test, train_size=valid_size, random_state=2021)

        device = self.device

        train_X = torch.tensor(train[:, :-1], dtype=torch.long).to(device)
        valid_X = torch.tensor(valid[:, :-1], dtype=torch.long).to(device)
        test_X = torch.tensor(test[:, :-1], dtype=torch.long).to(device)
        train_y = torch.tensor(train[:, -1], dtype=torch.float).unsqueeze(1).to(device)
        valid_y = torch.tensor(valid[:, -1], dtype=torch.float).unsqueeze(1).to(device)
        test_y = torch.tensor(test[:, -1], dtype=torch.float).unsqueeze(1).to(device)

        return field_dims, (train_X, train_y), (valid_X, valid_y), (test_X, test_y)


class CriteoDataset(Dataset):

    def __init__(self, file, read_part=True, sample_num=100000):
        super(CriteoDataset, self).__init__()

        names = ['label', 'I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11',
                 'I12', 'I13', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11',
                 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22',
                 'C23', 'C24', 'C25', 'C26']

        if read_part:
            data_df = pd.read_csv(file, sep='\t', header=None, names=names, nrows=sample_num)
        else:
            data_df = pd.read_csv(file, sep='\t', header=None, names=names)

        sparse_features = ['C' + str(i) for i in range(1, 27)]
        dense_features = ['I' + str(i) for i in range(1, 14)]
        features = sparse_features + dense_features

        # 缺失值填充
        data_df[sparse_features] = data_df[sparse_features].fillna('-1')
        data_df[dense_features] = data_df[dense_features].fillna(0)

        # 连续型特征等间隔分箱
        est = KBinsDiscretizer(n_bins=100, encode='ordinal', strategy='uniform')
        data_df[dense_features] = est.fit_transform(data_df[dense_features])

        # 离散型特征转换成连续数字，为了在与参数计算时使用索引的方式计算，而不是向量乘积
        data_df[features] = OrdinalEncoder().fit_transform(data_df[features])

        self.data = data_df[features + ['label']].values


class MovieLensDataset(Dataset):

    def __init__(self, file, read_part=True, sample_num=1000000, task='classification'):
        super(MovieLensDataset, self).__init__()

        dtype = {
            'userId': np.int32,
            'movieId': np.int32,
            'rating': np.float16,
        }
        if read_part:
            data_df = pd.read_csv(file, sep=',', dtype=dtype, nrows=sample_num)
        else:
            data_df = pd.read_csv(file, sep=',', dtype=dtype)
        data_df = data_df.drop(columns=['timestamp'])

        if task == 'classification':
            data_df['rating'] = data_df.apply(lambda x: 1 if x['rating'] > 3 else 0, axis=1).astype(np.int8)

        self.data = data_df.values


class AmazonBooksDataset(Dataset):

    def __init__(self, file, read_part=True, sample_num=100000, sequence_length=40):
        super(AmazonBooksDataset, self).__init__()

        if read_part:
            data_df = pd.read_csv(file, sep=',', nrows=sample_num)
        else:
            data_df = pd.read_csv(file, sep=',')

        data_df['hist_item_list'] = data_df.apply(lambda x: x['hist_item_list'].split('|'), axis=1)
        data_df['hist_cate_list'] = data_df.apply(lambda x: x['hist_cate_list'].split('|'), axis=1)

        # cate encoder
        cate_list = list(data_df['cateID'])
        data_df.apply(lambda x: cate_list.extend(x['hist_cate_list']), axis=1)
        cate_set = set(cate_list + ['0'])
        cate_encoder = LabelEncoder().fit(list(cate_set))
        self.cate_set = cate_encoder.transform(list(cate_set))

        # cate pad and transform
        hist_limit = sequence_length
        col = ['hist_cate_{}'.format(i) for i in range(hist_limit)]

        def deal(x):
            if len(x) > hist_limit:
                return pd.Series(x[-hist_limit:], index=col)
            else:
                pad = hist_limit - len(x)
                x = x + ['0' for _ in range(pad)]
                return pd.Series(x, index=col)

        cate_df = data_df['hist_cate_list'].apply(deal).join(data_df[['cateID']]).apply(cate_encoder.transform).join(
            data_df['label'])
        self.data = cate_df.values

    def train_valid_test_split(self, train_size=0.8, valid_size=0.1, test_size=0.1):
        field_dims = [self.data[:-1].max().astype(int) + 1]
        num_data = len(self.data)
        num_train = int(train_size * num_data)
        num_test = int(test_size * num_data)
        train = self.data[:num_train]
        valid = self.data[num_train: -num_test]
        test = self.data[-num_test:]

        device = self.device
        train_X = torch.tensor(train[:, :-1], dtype=torch.long).to(device)
        valid_X = torch.tensor(valid[:, :-1], dtype=torch.long).to(device)
        test_X = torch.tensor(test[:, :-1], dtype=torch.long).to(device)
        train_y = torch.tensor(train[:, -1], dtype=torch.float).unsqueeze(1).to(device)
        valid_y = torch.tensor(valid[:, -1], dtype=torch.float).unsqueeze(1).to(device)
        test_y = torch.tensor(test[:, -1], dtype=torch.float).unsqueeze(1).to(device)

        return field_dims, (train_X, train_y), (valid_X, valid_y), (test_X, test_y)


def create_dataset(dataset='criteo', read_part=True, sample_num=100000, task='classification', sequence_length=40, device=torch.device('cpu')):
    if dataset == 'criteo':
        return CriteoDataset('../dataset/criteo-100k.txt', read_part=read_part, sample_num=sample_num).to(device)
    elif dataset == 'movielens':
        return MovieLensDataset('../dataset/ml-latest-small-ratings.txt', read_part=read_part, sample_num=sample_num, task=task).to(device)
    elif dataset == 'amazon-books':
        return AmazonBooksDataset('../dataset/amazon-books-100k.txt', read_part=read_part, sample_num=sample_num, sequence_length=sequence_length).to(device)
    else:
        raise Exception('No such dataset!')


class EarlyStopper:

    def __init__(self, model, num_trials=50):
        self.num_trials = num_trials
        self.trial_counter = 0
        self.best_metric = -1e9
        self.best_state = deepcopy(model.state_dict())
        self.model = model

    def is_continuable(self, metric):
        # maximize metric
        if metric > self.best_metric:
            self.best_metric = metric
            self.trial_counter = 0
            self.best_state = deepcopy(self.model.state_dict())
            return True
        elif self.trial_counter + 1 < self.num_trials:
            self.trial_counter += 1
            return True
        else:
            return False


class BatchLoader:

    def __init__(self, X, y, batch_size=4096, shuffle=True):
        assert len(X) == len(y)

        self.batch_size = batch_size

        if shuffle:
            seq = list(range(len(X)))
            np.random.shuffle(seq)
            self.X = X[seq]
            self.y = y[seq]
        else:
            self.X = X
            self.y = y

    def __iter__(self):
        def iteration(X, y, batch_size):
            start = 0
            end = batch_size
            while start < len(X):
                yield X[start: end], y[start: end]
                start = end
                end += batch_size

        return iteration(self.X, self.y, self.batch_size)


class Trainer:

    def __init__(self, model, optimizer, criterion, batch_size=None, task='classification'):
        assert task in ['classification', 'regression']
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.batch_size = batch_size
        self.task = task

    def train(self, train_X, train_y, epoch=100, trials=None, valid_X=None, valid_y=None):
        if self.batch_size:
            train_loader = BatchLoader(train_X, train_y, self.batch_size)
        else:
            # 为了在 for b_x, b_y in train_loader 的时候统一
            train_loader = [[train_X, train_y]]

        if trials:
            early_stopper = EarlyStopper(self.model, trials)

        train_loss_list = []
        valid_loss_list = []

        for e in tqdm(range(epoch)):
            # train part
            self.model.train()
            train_loss_ = 0
            for b_x, b_y in train_loader:
                self.optimizer.zero_grad()
                pred_y = self.model(b_x)
                train_loss = self.criterion(pred_y, b_y)
                train_loss.backward()
                self.optimizer.step()

                train_loss_ += train_loss.detach() * len(b_x)

            train_loss_list.append(train_loss_ / len(train_X))

            # valid part
            if trials:
                valid_loss, valid_metric = self.test(valid_X, valid_y)
                valid_loss_list.append(valid_loss)
                if not early_stopper.is_continuable(valid_metric):
                    break

        if trials:
            self.model.load_state_dict(early_stopper.best_state)
            plt.plot(valid_loss_list, label='valid_loss')

        plt.plot(train_loss_list, label='train_loss')
        plt.legend()
        plt.show()

        print('train_loss: {:.5f} | train_metric: {:.5f}'.format(*self.test(train_X, train_y)))

        if trials:
            print('valid_loss: {:.5f} | valid_metric: {:.5f}'.format(*self.test(valid_X, valid_y)))

    def test(self, test_X, test_y):
        self.model.eval()
        with torch.no_grad():
            pred_y = self.model(test_X)
            test_loss = self.criterion(pred_y, test_y).detach()
        if self.task == 'classification':
            test_metric = metrics.roc_auc_score(test_y.cpu(), pred_y.cpu())
        elif self.task == 'regression':
            test_metric = -test_loss
        return test_loss, test_metric
