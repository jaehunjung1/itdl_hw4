import torch
import torch.nn as nn

import ipdb


class CharacterLSTM(nn.Module):
    def __init__(self, vocab_len):
        super(CharacterLSTM, self).__init__()
        self.vocab_len = vocab_len
        self.lstm = nn.LSTM(self.vocab_len, hidden_size=150,
                            bidirectional=True, batch_first=True,
                            dropout=0.3, num_layers=2)
        self.linear = nn.Linear(300, vocab_len)

    def forward(self, x):  # x : (batch_size, sentence_len)
        out = torch.arange(self.vocab_len).view(1, 1, -1).repeat((*x.size(), 1))
        out = (out == x.unsqueeze(2)).float()
        out, _ = self.lstm(out)
        out = self.linear(out)
        return out  # (64, 1, vocab_len)


if __name__ == '__main__':
    model = CharacterLSTM(86)
    batch = torch.zeros((64, 1)).long().random_(0, 86)
    out = model(batch)
    print(out.size())
