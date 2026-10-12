"""
TextCNN: slides filters of width 3/4/5 tokens over the text, so it learns
short local patterns. Max-pooling keeps the strongest
match anywhere in the text.

BiLSTM: reads the text token by token in both directions, so it can carry information
across the whole response (structure, ordering, how answers open and close).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import config


class TextCNN(nn.Module):
    def __init__(self, vocab_size, n_classes):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, config.EMB_DIM, padding_idx=0)
        self.convs = nn.ModuleList(
            nn.Conv1d(config.EMB_DIM, config.CNN_FILTERS, k) for k in config.CNN_KERNELS
        )
        self.drop = nn.Dropout(config.DROPOUT)
        self.fc = nn.Linear(config.CNN_FILTERS * len(config.CNN_KERNELS), n_classes)

    def forward(self, x, lengths=None):
        e = self.emb(x).transpose(1, 2)                          # (B, emb, T)
        mask = (x != 0).unsqueeze(1)                             # (B, 1, T)
        pooled = []
        for conv in self.convs:
            h = F.relu(conv(e))                                  # (B, filters, T-k+1)
            m = mask[:, :, : h.size(2)]
            h = h.masked_fill(~m, -1e4)                          # ignore windows starting on padding
            pooled.append(h.max(dim=2).values)
        return self.fc(self.drop(torch.cat(pooled, dim=1)))


class BiLSTM(nn.Module):
    def __init__(self, vocab_size, n_classes):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, config.EMB_DIM, padding_idx=0)
        self.lstm = nn.LSTM(
            config.EMB_DIM, config.LSTM_HIDDEN, num_layers=config.LSTM_LAYERS,
            batch_first=True, bidirectional=True,
            dropout=config.DROPOUT if config.LSTM_LAYERS > 1 else 0,
        )
        self.drop = nn.Dropout(config.DROPOUT)
        self.fc = nn.Linear(4 * config.LSTM_HIDDEN, n_classes)   # max-pool + mean-pool of 2*hidden

    def forward(self, x, lengths):
        e = self.drop(self.emb(x))
        packed = pack_padded_sequence(e, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=x.size(1))  # (B, T, 2H)
        mask = (torch.arange(x.size(1), device=x.device)[None, :] < lengths.to(x.device)[:, None]).unsqueeze(2)
        maxp = out.masked_fill(~mask, -1e4).max(dim=1).values
        meanp = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.fc(self.drop(torch.cat([maxp, meanp], dim=1)))


MODELS = {"CNN": TextCNN, "LSTM": BiLSTM}
