import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import time
import math

from model import CharacterLSTM

import ipdb

########### Change whether you would use GPU on this homework or not ############
USE_GPU = True
#################################################################################
if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

vocab = open('vocab.txt', encoding='utf-8').read().splitlines()
n_vocab = len(vocab)
torch.manual_seed(1)


# Change char to index.
def text2int(csv_file, dname, vocab):
    ret = []
    data = csv_file[dname].values
    for datum in data:
        for char in str(datum):
            idx = vocab.index(char)
            ret.append(idx)
    ret = np.array(ret)
    return ret


# Create dataset to automatically iterate.
class NewsDataset(Dataset):
    def __init__(self, csv_file, vocab):
        self.csv_file = pd.read_csv(csv_file, sep='|')
        self.vocab = vocab
        # self.len = len(self.csv_file)
        self.len = 2645
        self.x_data = torch.tensor(text2int(self.csv_file, 'x_data', self.vocab)[:169280]).view(-1, 64)
        self.y_data = torch.tensor(text2int(self.csv_file, 'y_data', self.vocab)[:169280]).view(-1, 64)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]


def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def train(dataset, model, optimizer, n_iters):
    model.to(device=device)
    model.train()
    start = time.time()
    print_every = 1
    criterion = nn.CrossEntropyLoss()
    for e in range(n_iters):
        for i, (x, y) in enumerate(dataset):
            x = x.to(device=device)
            y = y.to(device=device)
            model.zero_grad()
            output = model(x)  # output: (batch_size, sentence_len, vocab_len)
            loss = criterion(output.view(-1, vocab_len), y.view(-1))
            loss.backward()
            optimizer.step()
        if e % print_every == 0:
            print(f"Iteration {e}, {e / n_iters * 100} | {time_since(start)}, Loss: {loss}")
            print(test('W'))
        if e % 100 == 0:
            torch.save(model.state_dict(), f'./{e}.fng_pt.pt')


def test(start_letter):
    max_length = 1000
    with torch.no_grad():
        idx = vocab.index(start_letter)
        input_nparray = [idx]
        input_nparray = np.reshape(input_nparray, (1, len(input_nparray)))
        inputs = torch.tensor(input_nparray, device=device, dtype=torch.long)
        output_sen = start_letter
        for i in range(max_length):
            output = model(inputs).squeeze(0)
            topv, topi = output.topk(1)
            topi = topi[-1]
            letter = vocab[topi]
            output_sen += letter
            idx = vocab.index(letter)
            input_nparray = np.append(input_nparray, [idx])
            inputs = torch.tensor(input_nparray, device=device, dtype=torch.long).unsqueeze(0)
    return output_sen


if __name__ == '__main__':
    n_iters = 20
    vocab_len = len(vocab)
    dataset = NewsDataset(csv_file='data.csv', vocab=vocab)
    train_loader = DataLoader(dataset=dataset,
                              batch_size=64,
                              shuffle=False,
                              num_workers=1)
    model = CharacterLSTM(vocab_len)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=2e-16, weight_decay=0)
    train(train_loader, model, optimizer, n_iters)


