import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FSTTransBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(FSTTransBlock, self).__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.GELU()

    def forward(self, src):
        src2 = self.norm1(src)
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src = src + self.dropout2(self.linear2(self.dropout(self.activation(self.linear1(src2)))))
        return src

class FSTTrans(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_encoder_layers=4, num_decoder_layers=4, dim_feedforward=2048, dropout=0.1):
        super(FSTTrans, self).__init__()

        self.encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_encoder_layers)

        self.decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_decoder_layers)

        self.d_model = d_model
        self.nhead = nhead

        self.decoder_norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, 3)  # Output is 3D coordinates

    def forward(self, src, tgt):
        memory = self.transformer_encoder(src)
        output = self.transformer_decoder(tgt, memory)
        output = self.decoder_norm(output)
        output = self.out(output)
        return output
