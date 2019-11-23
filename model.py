import torch
import torch.nn as nn

import ipdb


class CharacterLSTM(nn.Module):
    def __init__(self, vocab_len):
        super(CharacterLSTM, self).__init__()
        self.embed = nn.Embedding(vocab_len, 50)
        self.lstm = nn.LSTM(50, hidden_size=75, bidirectional=True, batch_first=True, dropout=0.3, num_layers=2)
        self.linear = nn.Linear(150, vocab_len)

    def forward(self, x):
        out = self.embed(x)
        out, _ = self.lstm(out)
        out = self.linear(out)
        return out  # (64, 1, vocab_len)


if __name__ == '__main__':
    model = CharacterLSTM(86)
    batch = torch.zeros((64, 1)).long().random_(0, 86)
    out = model(batch)
    print(out.size())
