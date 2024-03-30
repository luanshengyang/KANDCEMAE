import random
import numpy as np
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.nn.init import xavier_normal_, uniform_, constant_
class PositionalEmbedding(nn.Module):

    def __init__(self, max_len, d_model):
        super(PositionalEmbedding, self).__init__()
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x):
        batch_size = x.size(0)
        return self.pe.weight.unsqueeze(0).repeat(batch_size, 1, 1)

class Attention(nn.Module):
    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn

class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)

class SublayerConnection(nn.Module):
    def __init__(self, size, enable_res_parameter, dropout=0.1):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
        self.enable = enable_res_parameter
        if enable_res_parameter:
            self.a = nn.Parameter(torch.tensor(1e-8))

    def forward(self, x, sublayer):
        if type(x) == list:
            return self.norm(x[1] + self.dropout(self.a * sublayer(x)))
        if not self.enable:
            return self.norm(x + self.dropout(sublayer(x)))
        else:
            return self.norm(x + self.dropout(self.a * sublayer(x)))

class PointWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ffn, dropout=0.1):
        super(PointWiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.linear2(self.activation(self.linear1(x))))

class TransformerBlock(nn.Module):
    """
    TRM layer
    """

    def __init__(self, d_model, attn_heads, d_ffn, enable_res_parameter, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attn = MultiHeadAttention(attn_heads, d_model, dropout)
        self.ffn = PointWiseFeedForward(d_model, d_ffn, dropout)
        self.skipconnect1 = SublayerConnection(d_model, enable_res_parameter, dropout)
        self.skipconnect2 = SublayerConnection(d_model, enable_res_parameter, dropout)

    def forward(self, x, mask):
        x = self.skipconnect1(x, lambda _x: self.attn.forward(_x, _x, _x, mask=mask))
        x = self.skipconnect2(x, self.ffn)
        return x

class Encoder_share(nn.Module):
    def __init__(self, args):
        super(Encoder_share, self).__init__()
        d_model = args.d_model
        attn_heads = args.attn_heads
        d_ffn = 4 * d_model
        layers = args.layers
        dropout = args.dropout
        enable_res_parameter = args.enable_res_parameter
        # TRMs
        self.TRMs = nn.ModuleList(
            [TransformerBlock(d_model, attn_heads, d_ffn, enable_res_parameter, dropout) for i in range(8)])

    def forward(self, x):
        for TRM in self.TRMs:
            x = TRM(x, mask=None)
        return x

class Encoder_I(nn.Module):
    def __init__(self, args):
        super(Encoder_I, self).__init__()
        d_model = args.d_model
        attn_heads = args.attn_heads
        d_ffn = 4 * d_model                         # 4×152
        layers = args.layers
        dropout = args.dropout
        enable_res_parameter = args.enable_res_parameter
        # TRMs
        self.TRMs = nn.ModuleList(
            [TransformerBlock(d_model, attn_heads, d_ffn, enable_res_parameter, dropout) for i in range(3)])

    def forward(self, x):
        for TRM in self.TRMs:
            x = TRM(x, mask=None)
        return x

class Encoder_Q(nn.Module):
    def __init__(self, args):
        super(Encoder_Q, self).__init__()
        d_model = args.d_model
        attn_heads = args.attn_heads
        d_ffn = 4 * d_model                         # 4×152
        layers = args.layers
        dropout = args.dropout
        enable_res_parameter = args.enable_res_parameter
        # TRMs
        self.TRMs = nn.ModuleList(
            [TransformerBlock(d_model, attn_heads, d_ffn, enable_res_parameter, dropout) for i in range(3)])

    def forward(self, x):
        for TRM in self.TRMs:
            x = TRM(x, mask=None)
        return x
class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        decoder_d_model = int(args.d_model/2)
        attn_heads = args.attn_heads
        d_ffn = 4 * decoder_d_model
        layers = args.layers
        dropout = args.dropout
        enable_res_parameter = args.enable_res_parameter
        # TRMs
        self.TRMs = nn.ModuleList(
            [TransformerBlock(decoder_d_model*2, attn_heads, d_ffn, enable_res_parameter, dropout) for i in range(3)])

    def forward(self, x):
        for TRM in self.TRMs:
            x = TRM(x, mask=None)
        return x

class DCEMAE(nn.Module):
    def __init__(self, args):
        super(DCEMAE, self).__init__()
        d_model = args.d_model                                      # d_model = 152
        self.device = args.device
        self.data_shape = args.data_shape                           # data_shape = (128, 2)
        self.max_len = int(self.data_shape[0] / args.wave_length)   # wave_length = 8
        norm_layer = nn.LayerNorm
        self.mask_len = int(args.mask_ratio * self.max_len)
        self.position = PositionalEmbedding(self.max_len, d_model)
        seq_len = 16
        half_d_model = int(d_model / 2)
        self.norm = norm_layer(d_model)

        self.mask_token_I = nn.Parameter(torch.randn(d_model, ))
        self.mask_token_Q = nn.Parameter(torch.randn(d_model, ))
        self.input_proj_I = nn.Conv1d(1, d_model, kernel_size=args.wave_length,stride=args.wave_length)
        self.input_proj_Q = nn.Conv1d(1, d_model, kernel_size=args.wave_length,stride=args.wave_length)

        self.encoder_share = Encoder_share(args)
        self.encoder_I = Encoder_I(args)
        self.encoder_Q = Encoder_Q(args)
        self.decoder = Decoder(args)
        self.decoder_norm_I = norm_layer(d_model)
        self.decoder_norm_Q = norm_layer(d_model)
        self.decoder_pred_I = nn.Linear(d_model, d_model, bias=True, )
        self.decoder_pred_Q = nn.Linear(d_model, d_model, bias=True, )

        self.predict_head_1 = nn.Linear(d_model, 100)
        self.predict_head_2 = nn.Linear(100, 50)
        self.predict_head_3 = nn.Linear(50, args.num_class)
        self.apply(self._init_weights)

        self.output_proj_I_1 = nn.Conv1d(d_model, 1, kernel_size=1)
        self.output_proj_Q_1 = nn.Conv1d(d_model, 1, kernel_size=1)
        self.output_proj_I_2 = nn.Conv1d(seq_len, 128, kernel_size=1)
        self.output_proj_Q_2 = nn.Conv1d(seq_len, 128, kernel_size=1)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0.1)

    def forward(self, x):       # x_input = (Bs, 128, 2)
        x_I = x[:, :, 0]
        x_Q = x[:, :, 1]
        x_I_exp = x_I.unsqueeze(dim=2)
        x_Q_exp = x_Q.unsqueeze(dim=2)
        x_I_1 = self.input_proj_I(x_I_exp.transpose(1, 2)).transpose(1, 2).contiguous()
        x_Q_1 = self.input_proj_Q(x_Q_exp.transpose(1, 2)).transpose(1, 2).contiguous()
        x_pos_bed_I = x_I_1 + self.position(x_I_1)
        x_pos_bed_Q = x_Q_1 + self.position(x_Q_1)

        x_En_output_I = self.encoder_I(x_pos_bed_I)
        x_En_output_Q = self.encoder_Q(x_pos_bed_Q)

        x_En_share_input_IQ = torch.cat([x_En_output_I, x_En_output_Q], dim=1)
        x_En_share_output_IQ = self.encoder_share(x_En_share_input_IQ)
        W1 = int(x_En_share_output_IQ.shape[1] / 2)
        x_En_share_output_I = x_En_share_output_IQ[:, 0:W1, :]
        x_En_share_output_Q = x_En_share_output_IQ[:, W1:2 * W1, :]
        x_En_share_output_IQ = torch.cat([x_En_share_output_I, x_En_share_output_Q], dim=1)

        x_En_share_output_IQ_mean = torch.mean(x_En_share_output_IQ, dim=1)

        x_MLP_1 = self.predict_head_1(x_En_share_output_IQ_mean)
        x_MLP_2 = self.predict_head_2(x_MLP_1)
        x_MLP_3 = self.predict_head_3(x_MLP_2)
        return x_MLP_3

