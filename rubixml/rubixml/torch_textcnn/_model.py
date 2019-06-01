import os
import sys
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch
import torchtext
from torchtext import datasets
from torchtext.data import (
    Field,
    TabularDataset,
    Iterator,
    BucketIterator,
    Pipeline
)
import numpy as np
import json

from collections import Counter

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from ._network import TextCNN


class TextCNNSentimentClassifier(object):
    """ A typical convolutional neural network for text classification
    Paper: https://arxiv.org/pdf/1408.5882.pdf
    """

    def __init__(self, embed_dim, lr, dropout, use_gpu=False):
        """
        Args:
            embed_dim: integer, the dimension of word embedding
            lr: float, learning rate
            dropout: float, probability of an element to be zeroed.
            use_gpu: boolean, whether use GPU
        """
        self.embed_dim = embed_dim
        self.lr = lr
        self.dropout = dropout
        self.use_gpu = use_gpu

        self.text_field = Field(sequential=True, lower=True, fix_length=50)
        self.label_field = Field(sequential=False, is_target=True)

        self.network = None
        return

    def _process_data(self, filepath, train_dev_ratio):
        """ preprocess dataset

        Args:
            filepath: string, the path of dataset
            train_dev_ratio: a float, the ratio to split train and dev dataset

        Returns:
            A tuple of torchtext.data.Dataset objects: (train, dev)
        """
        train, dev = TabularDataset(
            path=filepath,
            format='csv',
            fields=[('text', self.text_field), ('label', self.label_field)],
            csv_reader_params=dict(delimiter='\t')
        ).split(split_ratio=train_dev_ratio)

        train_words = list(map(lambda x: len(x.text), train.examples))
        train_labels = list(map(lambda x: int(x.label), train.examples))
        dev_words = list(map(lambda x: len(x.text), dev.examples))
        dev_labels = list(map(lambda x: int(x.label), dev.examples))

        print('----------------------------------------------------------')
        print('train: min words={}, max words={}, counter={}'.format(
            min(train_words), max(train_words), str(Counter(train_labels))))
        print('dev: min words={}, max words={}, counter={}'.format(
            min(dev_words), max(dev_words), str(Counter(dev_labels))))
        print('----------------------------------------------------------')
        print('\n')

        return train, dev

    def _build_network(self):
        """ build specific TextCNN network
        """
        vocab_size = len(self.text_field.vocab)
        if vocab_size == 0:
            raise Exception('Please call fit() function first!')

        network_params = {
            'vocab_size': vocab_size,
            'embed_dim': self.embed_dim,
            'class_num': 1,
            'kernel_num': 100,
            'kernel_sizes': [3, 4, 5],
            'dropout': self.dropout,
            'static': False,
        }

        self.network = TextCNN(**network_params)
        return

    def fit(self, filepath, train_dev_ratio=0.8, batch_size=64, nepoch=10):
        """ Feed training data to train the model

        Args:
            filepath: a string, the path of dataset
            train_dev_ratio: a float, the ratio to split train and dev dataset
            batch_size: a integer, the size of batch when training
            nepoch: a integer, the number of training epochs

        TODO:
            1) support early stopping
            2) support customized delimiter
            3) support callback function
            4) support fit_generator
        """
        train, dev = self._process_data(filepath, train_dev_ratio)

        self.text_field.build_vocab(train, vectors="glove.6B.50d")
        self.label_field.build_vocab(train)

        self._build_network()

        train_iter = Iterator(train, batch_size=batch_size, shuffle=True)
        dev_iter = Iterator(dev, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr)
        loss_fn = nn.BCELoss()

        self.network.train()
        best_acc = 0
        for epoch in range(1, nepoch + 1):
            for i, batch in enumerate(train_iter):
                feature, target = batch.text, batch.label
                target = target.type(torch.FloatTensor)
                feature.data.t_(), target.data.sub_(1)
                optimizer.zero_grad()

                y_pred = self.network(feature).reshape(-1)
                loss = loss_fn(y_pred, target)
                loss.backward()
                optimizer.step()

                label_pred = (np.array(y_pred.data) > 0.5).astype(int)
                label_true = np.array(target)
                train_acc = accuracy_score(label_true, label_pred)
                output_str = '\rEpoch:{} batch:{} loss:{:.6f} acc:{:.2f}'
                sys.stdout.write(output_str.format(epoch,
                                                   i,
                                                   loss.item(),
                                                   train_acc))

            dev_acc = self.evaluate(dev_iter)
            if dev_acc > best_acc:
                best_acc = dev_acc
                print('Saving best model, acc: {:.4f}\n'.format(best_acc))
                self._save_weights(self.network)

        return

    def evaluate(self, dev_data):
        """ evaluate the dev dataset
        Args:
            dev_data: torchtext.data.Iterator or torchtext.data.Dataset

        Returns:
            a tuple of (accuracy, precision, recall, f1_score)
        """
        if isinstance(dev_data, Iterator):
            dev_iter = dev_data
        else:
            dev_iter = Iterator(dev_data, batch_size=32)

        self.network.eval()
        label_pred, label_true = [], []
        for batch in dev_iter:
            feature, target = batch.text, batch.label
            target = target.type(torch.FloatTensor)
            # since the label is {unk:0, 0: 1, 1: 2}, need subtrct 1
            feature.data.t_(), target.data.sub_(1)
            y_pred = self.network(feature)
            y_pred = y_pred.reshape(-1)
            label_pred += list((np.array(y_pred.data) > 0.5).astype(int))
            label_true += list(np.array(target))

        acc = accuracy_score(label_true, label_pred)
        p = precision_score(label_true, label_pred)
        r = recall_score(label_true, label_pred)
        f1 = f1_score(label_true, label_pred)
        output_str = '\nEval - acc:{:.2f} p:{:.2f} r:{:.2f} f1:{:.2f} \n'
        print(output_str.format(acc, p, r, f1))
        return acc, p, r, f1

    def predict_prob(self, sentences=[]):
        """ Predict the probability of the sentiment between 0 and 1

        Args:
            sentences: a list of strings, the sentences used to predict

        Returns:
            a list of floats, the predicted probability
        """

        self.network.eval()
        sentences = [self.text_field.preprocess(sent) for sent in sentences]
        sentences = self.text_field.pad(sentences)
        sentences = [[self.text_field.vocab.stoi[word] for word in sent]
                     for sent in sentences]

        X = torch.tensor(sentences)
        X = torch.autograd.Variable(X)
        if self.use_gpu:
            X = X.cuda()

        y_pred = self.network(X)
        y_pred = np.array(y_pred.data).reshape(-1)
        return y_pred

    def predict(self, sentences):
        """ Predict the label of the sentiment: 0 or 1

        Args:
            sentences: a list of strings, the sentences used to predict

        Returns:
            a list of int
        """
        prob = self.predict_prob(sentences)
        label = (prob > 0.5).astype(int)
        return label

    def _save_weights(self, network, save_dir='.textcnn', save_prefix='best'):
        """ save the weight of network
        Args:
            network: torch.nn.Module
            save_dir: string, the path of weights directory
            save_prefix: string, the prefix of weights filename
        """
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        save_prefix = os.path.join(save_dir, save_prefix)
        save_path = '{}_network_weights.pt'.format(save_prefix)
        torch.save(network.state_dict(), save_path)
        return

    def use_best_model(self, save_dir='.textcnn', save_prefix='best'):
        """ overwrite the current model weights with the best model weights

        Args:
            save_dir: string, the path of weights directory
            save_prefix: string, the prefix of weights filename
        """
        save_prefix = os.path.join(save_dir, save_prefix)
        save_path = '{}_network_weights.pt'.format(save_prefix)
        self.network.load_state_dict(torch.load(save_path))
        return
