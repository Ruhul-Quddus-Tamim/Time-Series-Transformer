import torch
from torch import nn
import numpy as np
import logging

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.logger = logging.getLogger(__name__)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return x

class TimeSeriesModel(nn.Module):
    def __init__(self, seq_length, target_length, d_model, num_heads, num_layers, dropout=0.1):
        super(TimeSeriesModel, self).__init__()
        self.logger = logging.getLogger(__name__)
        self.seq_length = seq_length
        self.target_length = target_length
        self.d_model = d_model
        self.input_proj = nn.Linear(1, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_length)
        self.pos_decoder = PositionalEncoding(d_model, max_len=target_length)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=128, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=128, dropout=dropout
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.conv = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(d_model, 1)

    def forward(self, src, tgt, tgt_mask=None):
        src = self.input_proj(src)
        src = self.pos_encoder(src.permute(1, 0, 2))
        src_conv = self.conv(src.permute(1, 2, 0))
        src_conv = self.dropout(src_conv)
        src = src + src_conv.permute(2, 0, 1)
        memory = self.transformer_encoder(src)

        tgt = self.input_proj(tgt)
        tgt = self.pos_decoder(tgt.permute(1, 0, 2))
        tgt_conv = self.conv(tgt.permute(1, 2, 0))
        tgt_conv = self.dropout(tgt_conv)
        tgt = tgt + tgt_conv.permute(2, 0, 1)

        output = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask)
        output = self.output_proj(output)
        output = output.permute(1, 0, 2)
        return output

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        return mask