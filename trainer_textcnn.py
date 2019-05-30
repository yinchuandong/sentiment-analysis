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
import pandas as pd
import numpy as np
import json

from collections import Counter

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
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

        self.class_num = class_num
        self.static = static

        V = embed_num
        D = embed_dim
        C = class_num
        Ci = 1  # input channel
        Co = kernel_num
        Ks = kernel_sizes

        self.embed = nn.Embedding(V, D)
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(len(Ks) * Co, C)
        return

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
        if self.class_num > 1:
            y_pred = F.softmax(logit)
        else:
            y_pred = F.sigmoid(logit)
        return y_pred


def eval(data_iter, model):
    model.eval()
    label_pred, label_true = [], []
    for batch in data_iter:
        feature, target = batch.text, batch.label
        target = target.type(torch.FloatTensor)
        # since the label is {unk:0, 0: 1, 1: 2}, need subtrct 1
        feature.data.t_(), target.data.sub_(1)
        y_pred = model(feature)
        y_pred = y_pred.reshape(-1)
        label_pred += list((np.array(y_pred.data) > 0.5).astype(int))
        label_true += list(np.array(target))
    acc = accuracy_score(label_true,label_pred)
    output_str = '\nEvaluation -  dev acc: {:.2f} \n'
    print(output_str.format(acc))
    return acc


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
    y_pred = model(x)
    y_pred = np.array(y_pred.data).reshape(-1)
    return y_pred


def save(model, save_dir, save_prefix, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_steps_{}.pt'.format(save_prefix, steps)
    torch.save(model.state_dict(), save_path)


# %%

TEXT = Field(sequential=True, lower=True, fix_length=50)
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

train_lengths = list(map(lambda x: len(x.text), train.examples))
train_labels = list(map(lambda x: int(x.label), train.examples))
test_lengths = list(map(lambda x: len(x.text), test.examples))
test_labels = list(map(lambda x: int(x.label), test.examples))

print('train:', min(train_lengths), max(train_lengths), Counter(train_labels))
print('test:', min(test_lengths), max(test_lengths), Counter(test_labels))
# %%
args = {
    'embed_num': len(TEXT.vocab),
    'embed_dim': 50,
    'class_num': 1,
    'kernel_num': 100,
    'kernel_sizes': [3, 4, 5],
    'dropout': 0.5,
    'static': False,
}

# print(json.dumps(args, indent=2))


epochs = 15
batch_size = 64
steps = 0
best_acc = 0
last_step = 0
log_interval = 10
early_stopping = 30


train_iter = Iterator(train, batch_size=batch_size, shuffle=True)
test_iter = Iterator(test, batch_size=batch_size, shuffle=True)

# %%
model = TextCNN(**args)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.BCELoss()

model.train()

for epoch in range(1, epochs + 1):
    for batch in train_iter:
        feature, target = batch.text, batch.label
        target = target.type(torch.FloatTensor)
        feature.data.t_(), target.data.sub_(1)
        optimizer.zero_grad()
        y_pred = model(feature).reshape(-1)
        loss = loss_fn(y_pred, target)
        loss.backward()
        optimizer.step()
        steps += 1

        label_pred = (np.array(y_pred.data) > 0.5).astype(int)
        label_true = np.array(target)
        train_acc = accuracy_score(label_true,label_pred)
        output_str = '\rEpoch: {} - Batch: {} - loss: {:.6f}  train acc: {:.2f}'
        sys.stdout.write(output_str.format(epoch,
                                           steps,
                                           loss.item(),
                                           train_acc))

    dev_acc = eval(test_iter, model)
    if dev_acc > best_acc:
        best_acc = dev_acc
        last_step = steps
        print('Saving best model, acc: {:.4f}%\n'.format(best_acc))
        save(model, 'tmp-torch-textcnn-model', 'best', 0)


# %%

print('-----------------------------------------------------------------------')
# model.load_state_dict(torch.load('tmp-torch-textcnn-model/best_steps_0.pt'))
# raw_text = 'how'
# raw_text = 'Wow... Loved this place.'
# raw_text = 'Crust is not good.'
# raw_text = 'Not tasty and the texture was just nasty.'
# raw_text = 'Stopped by during the late May bank holiday off Rick Steve recommendation and loved it.'
raw_text = 'There was a warm feeling with the service and I felt like their guest for a special treat.'
pred_label = predict(raw_text, model, TEXT, LABEL)
print('predict:', pred_label)


# %%
# t1 = TEXT.preprocess(raw_text)
# t1
#
# t2 = TEXT.pad([t1])
# TEXT.vocab.stoi['<pad>']
# %%
