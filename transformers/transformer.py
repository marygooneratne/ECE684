import torch
import torch.nn as nn
import torch.optim as optim
import random
import string


class Transformer(nn.Module):
    def __init__(
        self,
        embedding_size,
        src_pad_idx,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        forward_expansion,
        dropout,
        max_len,
        device,
    ):
        super(Transformer, self).__init__()
        alphasize = 27
        self.src_word_embedding = nn.Embedding(alphasize, embedding_size)
        self.src_position_embedding = nn.Embedding(max_len, embedding_size)
        self.trg_word_embedding = nn.Embedding(alphasize, embedding_size)
        self.trg_position_embedding = nn.Embedding(max_len, embedding_size)

        self.device = device
        self.transformer = nn.Transformer(
            embedding_size,
            num_heads,
            num_encoder_layers,
            num_decoder_layers,
            forward_expansion,
            dropout,
        )
        self.fc_out = nn.Linear(embedding_size, alphasize)
        self.dropout = nn.Dropout(dropout)
        self.src_pad_idx = src_pad_idx

    def make_mask(self, src):
        mask = src.transpose(0, 1) == self.src_pad_idx

        return mask.to(self.device)

    def forward(self, src, trg):
        src_len, N = src.shape
        trg_len, N = trg.shape

        src_pos = (
            torch.arange(0, src_len)
            .unsqueeze(1)
            .expand(src_len, N)
            .to(self.device)
        )
        trg_pos = (
            torch.arange(0, trg_len)
            .unsqueeze(1)
            .expand(trg_len, N)
            .to(self.device)
        )

        embed_src = self.dropout(
            (self.src_word_embedding(src) +
             self.src_position_embedding(src_pos))
        )

        embed_trg = self.dropout(
            (self.trg_word_embedding(trg) +
             self.trg_position_embedding(trg_pos))
        )
        src_padding_mask = self.make_mask(src)
        trg_mask = self.transformer.generate_square_subsequent_mask(trg_seq_length).to(
            self.device
        )
        out = self.transformer(
            embed_src,
            embed_trg,
            src_key_padding_mask=src_padding_mask,
            tgt_mask=trg_mask,
        )
        out = self.fc_out(out)
        return out
