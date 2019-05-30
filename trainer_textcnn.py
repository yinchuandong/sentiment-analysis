import os
import sys
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch
import torchtext
from torchtext import datasets
from torchtext.data import Field, TabularDataset, Iterator, BucketIterator, Pipeline
import pandas as pd
import numpy as np
import json

from collections import Counter

# %%
class TextCNN(nn.Module):

    def __init__(self,
                 embed_num,
                 embed_dim,
                 class_num,
                 kernel_num,
                 kernel_sizes,
                 dropout,
                 static=True):
        super(TextCNN, self).__init__()

        self.static = static

        V = embed_num
        D = embed_dim
        C = class_num
        Ci = 1  # input channel
        Co = kernel_num
        Ks = kernel_sizes

        self.embed = nn.Embedding(V, D)
        # self.convs1 = [nn.Conv2d(Ci, Co, (K, D)) for K in Ks]
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])

        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(len(Ks) * Co, C)

    def forward(self, x):
        x = self.embed(x)  # (N, W, D)

        if not self.static:
            x = Variable(x)

        x = x.unsqueeze(1)  # (N, Ci, W, D)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]

        x = torch.cat(x, 1)
        x = self.dropout(x)  # (N, len(Ks)*Co)
        logit = self.fc1(x)  # (N, C)
        return logit


def eval(data_iter, model):
    model.eval()
    corrects, avg_loss = 0, 0
    for batch in data_iter:
        feature, target = batch.text, batch.label
        # since the label is {unk:0, 0: 1, 1: 2}, need subtrct 1
        feature.data.t_(), target.data.sub_(1)
        logits = model(feature)
        loss = F.cross_entropy(logits, target)
        avg_loss += loss.item()
        corrects += (torch.max(logits, 1)
                     [1].view(target.size()).data == target.data).sum()
    size = len(data_iter.dataset)
    avg_loss /= size
    accuracy = 100.0 * corrects / size
    output_str = '\nEvaluation - loss: {:.6f}  dev acc: {:.2f}% ({}/{}) \n'
    print(output_str.format(avg_loss,
                            accuracy,
                            corrects,
                            size))
    return accuracy


def predict(text, model, text_field, label_feild, cuda_flag=False):
    assert isinstance(text, str)
    model.eval()
    # text = text_field.tokenize(text)
    text = text_field.preprocess(text)
    text = text_field.pad([text])[0]
    text = [[text_field.vocab.stoi[x] for x in text]]
    x = torch.tensor(text)
    x = torch.autograd.Variable(x)
    if cuda_flag:
        x = x.cuda()
    print(x)
    output = model(x)
    _, predicted = torch.max(output, 1)
    # return label_feild.vocab.itos[predicted.data[0][0]+1]
    return label_feild.vocab.itos[predicted.data[0] + 1]


def save(model, save_dir, save_prefix, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_steps_{}.pt'.format(save_prefix, steps)
    torch.save(model.state_dict(), save_path)


# %%

TEXT = Field(sequential=True, lower=True)
LABEL = Field(sequential=False, is_target=True)

train, test = TabularDataset(
    path='./raw_data/train.txt',
    format='csv',
    fields=[('text', TEXT), ('label', LABEL)],
    csv_reader_params=dict(delimiter='\t')).split(split_ratio=0.8)

TEXT.build_vocab(train, vectors="glove.6B.50d")
LABEL.build_vocab(train)

# len(TEXT.vocab)
# print(TEXT.vocab.itos)
# TEXT.vocab.stoi
# TEXT.vocab.vectors
#
LABEL.vocab.stoi
LABEL.vocab.itos

train_labels = list(map(lambda x: int(x.label), train.examples))
test_labels = list(map(lambda x: int(x.label), test.examples))

print('train:', Counter(train_labels))
print('test: ', Counter(test_labels))
# %%
args = {
    'embed_num': len(TEXT.vocab),
    'embed_dim': 50,
    'class_num': len(LABEL.vocab) - 1,
    'kernel_num': 100,
    'kernel_sizes': [3, 4, 5],
    'dropout': 0.5,
    'static': True,
}

# print(json.dumps(args, indent=2))


epochs = 30
batch_size = 64
steps = 0
best_acc = 0
last_step = 0
log_interval = 10
early_stopping = 30


train_iter = Iterator(train, batch_size=batch_size)
test_iter = Iterator(test, batch_size=batch_size)

# %%
model = TextCNN(**args)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

model.train()

for epoch in range(1, epochs + 1):
    for batch in train_iter:
        feature, target = batch.text, batch.label
        feature.data.t_(), target.data.sub_(1)

        optimizer.zero_grad()
        logits = model(feature)
        loss = F.cross_entropy(logits, target)
        loss.backward()
        optimizer.step()
        steps += 1

        corrects = (torch.max(logits, 1)[1].view(
            target.size()).data == target.data).sum()
        train_acc = 100.0 * corrects / batch.batch_size
        output_str = '\rEpoch: {} - Batch: {} - loss: {:.6f}  train acc: {:.2f}% ({}/{})'
        sys.stdout.write(output_str.format(epoch,
                                           steps,
                                           loss.item(),
                                           train_acc,
                                           corrects,
                                           batch.batch_size))

    dev_acc = eval(test_iter, model)
    if dev_acc > best_acc:
        best_acc = dev_acc
        last_step = steps
        print('Saving best model, acc: {:.4f}%\n'.format(best_acc))
        save(model, 'tmp-torch-textcnn-model', 'best', 0)
    else:
        if steps - last_step >= early_stopping:
            print('\nearly stop by {} steps, acc: {:.4f}%'.format(
                early_stopping, best_acc))


# %%

print('-----------------------------------------------------------------------')
model.load_state_dict(torch.load('tmp-torch-textcnn-model/best_steps_0.pt'))
raw_text = 'how are you going hello hello hello'
predict(raw_text, model, TEXT, LABEL)


# %%
# t1 = TEXT.preprocess(raw_text)
# t1
#
# t2 = TEXT.pad([t1])
# TEXT.vocab.stoi['<pad>']
# %%
