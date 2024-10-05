import torch
import torch.nn.functional as F
import random
import numpy as np
import torch.nn as nn
from torch.nn.init import xavier_normal_, uniform_, constant_
import math

class PositionalEmbedding(nn.Module):

    def __init__(self, max_len, d_model):
        super(PositionalEmbedding, self).__init__()

        # 计算(max_len,d_model)的Embedding矩阵
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x):
        batch_size = x.size(0)
        # 升维，将(batchsize)变成(batchsize,max_len,d_model)大小的PositionalEmbedding
        return self.pe.weight.unsqueeze(0).repeat(batch_size, 1, 1)

class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """
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
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        # self.output_linear = nn.Linear(d_model, d_model)
        self.output_linear = KANModel([d_model, d_model])
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        x = self.output_linear(x)
        return x


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    """

    def __init__(self, size, enable_res_parameter, dropout=0.1):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
        self.enable = enable_res_parameter
        if enable_res_parameter:
            self.a = nn.Parameter(torch.tensor(1e-8))

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        if type(x) == list:
            return self.norm(x[1] + self.dropout(self.a * sublayer(x)))
        if not self.enable:
            return self.norm(x + self.dropout(sublayer(x)))
        else:
            return self.norm(x + self.dropout(self.a * sublayer(x)))

class PointWiseFeedForward(nn.Module):
    """
    FFN implement
    """

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

class CrossAttnTRMBlock(nn.Module):
    def __init__(self, d_model, attn_heads, d_ffn, enable_res_parameter, dropout=0.1):
        super(CrossAttnTRMBlock, self).__init__()
        self.attn = MultiHeadAttention(attn_heads, d_model, dropout)
        self.ffn = PointWiseFeedForward(d_model, d_ffn, dropout)
        self.skipconnect1 = SublayerConnection(d_model, enable_res_parameter, dropout)
        self.skipconnect2 = SublayerConnection(d_model, enable_res_parameter, dropout)

    def forward(self, rep_visible, rep_mask_token, mask=None):
        x = [rep_visible, rep_mask_token]
        x = self.skipconnect1(x, lambda _x: self.attn.forward(_x[1], _x[0], _x[0], mask=mask))
        x = self.skipconnect2(x, self.ffn)
        return x

class KANLinear(torch.nn.Module):
    def __init__(
            self,
            in_features,
            out_features,
            grid_size=5,
            spline_order=3,
            scale_noise=0.1,
            scale_base=1.0,
            scale_spline=1.0,
            enable_standalone_scale_spline=True,
            base_activation=torch.nn.SiLU,
            grid_eps=0.02,
            grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                    torch.arange(-spline_order, grid_size + spline_order + 1) * h
                    + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                    (
                            torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                            - 1 / 2
                    )
                    * self.scale_noise
                    / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order: -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                            (x - grid[:, : -(k + 1)])
                            / (grid[:, k:-1] - grid[:, : -(k + 1)])
                            * bases[:, :, :-1]
                    ) + (
                            (grid[:, k + 1:] - x)
                            / (grid[:, k + 1:] - grid[:, 1:(-k)])
                            * bases[:, :, 1:]
                    )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.view(-1, self.in_features)

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        output = base_output + spline_output

        output = output.view(*original_shape[:-1], self.out_features)
        return output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
                torch.arange(
                    self.grid_size + 1, dtype=torch.float32, device=x.device
                ).unsqueeze(1)
                * uniform_step
                + x_sorted[0]
                - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
                regularize_activation * regularization_loss_activation
                + regularize_entropy * regularization_loss_entropy
        )

class KANModel(torch.nn.Module):
    def __init__(
            self,
            layers_hidden,
            grid_size=5,
            spline_order=3,
            scale_noise=0.1,
            scale_base=1.0,
            scale_spline=1.0,
            base_activation=torch.nn.SiLU,
            grid_eps=0.02,
            grid_range=[-1, 1],
    ):
        super(KANModel, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order

        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )

    def forward(self, x: torch.Tensor, update_grid=False):
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )

class Encoder_share(nn.Module):
    def __init__(self, args):
        super(Encoder_share, self).__init__()
        d_model = args.d_model
        attn_heads = args.attn_heads
        d_ffn = 4 * d_model                         # 4×152
        layers = args.layers
        dropout = args.dropout
        enable_res_parameter = args.enable_res_parameter
        # TRMs
        self.TRMs = nn.ModuleList(
            [TransformerBlock(d_model, attn_heads, d_ffn, enable_res_parameter, dropout) for i in range(layers)])  # layers=8

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
            [TransformerBlock(d_model, attn_heads, d_ffn, enable_res_parameter, dropout) for i in range(3)])  # layers=8

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
            [TransformerBlock(d_model, attn_heads, d_ffn, enable_res_parameter, dropout) for i in range(3)])  # layers=8

    def forward(self, x):
        for TRM in self.TRMs:
            x = TRM(x, mask=None)
        return x
class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        decoder_d_model = int(args.d_model/2)
        attn_heads = args.attn_heads
        d_ffn = 4 * decoder_d_model                         # 4×152
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

class Tokenizer(nn.Module):
    def __init__(self, rep_dim, vocab_size):
        super(Tokenizer, self).__init__()
        self.center = nn.Linear(rep_dim, vocab_size)

    def forward(self, x):
        bs, length, dim = x.shape
        shapppe = x.view(-1, dim)
        probs = self.center(x.view(-1, dim))
        ret = F.gumbel_softmax(probs)
        indexes = ret.max(-1, keepdim=True)[1]
        return indexes.view(bs, length)

class Regressor(nn.Module):
    def __init__(self, d_model, attn_heads, d_ffn, enable_res_parameter, layers):
        super(Regressor, self).__init__()
        self.layers = nn.ModuleList(
            [CrossAttnTRMBlock(d_model, attn_heads, d_ffn, enable_res_parameter) for i in range(2)])

    def forward(self, rep_visible, rep_mask_token):
        for TRM in self.layers:
            rep_mask_token = TRM(rep_visible, rep_mask_token)
        return rep_mask_token

class OurMAE(nn.Module):
    def __init__(self, args):
        super(OurMAE, self).__init__()
        d_model = args.d_model
        self.linear_proba = True
        self.momentum = args.momentum
        self.device = args.device
        self.data_shape = args.data_shape
        self.max_len = int(self.data_shape[0] / args.wave_length)
        norm_layer = nn.LayerNorm
        # print(self.data_shape[0])
        # print(self.max_len)Compute the positional encodings once in log space.
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

        self.predict_head_1 = KANModel([d_model, 100])
        self.predict_head_2 = KANModel([100, 50])
        self.predict_head_3 = KANModel([50, args.num_class])

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

    def momentum_update(self):
        with torch.no_grad():
            for (param_a, param_b) in zip(self.encoder.parameters(), self.momentum_encoder.parameters()):
                param_b.data = self.momentum * param_b.data + (1 - self.momentum) * param_a.data

    def pretrain_forward(self, x_input):
        # x_input.shape=256,128,2
        x_I = x_input[:,:,0]
        x_Q = x_input[:,:,1]
        x_I_exp = x_I.unsqueeze(dim=2)  # 256,128,1
        x_Q_exp = x_Q.unsqueeze(dim=2)  # 256,128,1
        x_I_1 = self.input_proj_I(x_I_exp.transpose(1, 2)).transpose(1, 2).contiguous()     # 256,16,152
        x_Q_1 = self.input_proj_Q(x_Q_exp.transpose(1, 2)).transpose(1, 2).contiguous()     # 256,16,152
        x_pos_bed_I = x_I_1 + self.position(x_I_1)
        x_pos_bed_Q = x_Q_1 + self.position(x_Q_1)
        rep_mask_token_I = self.mask_token_I.repeat(x_pos_bed_I.shape[0], x_pos_bed_I.shape[1], 1) + self.position(x_I_1)
        rep_mask_token_Q = self.mask_token_Q.repeat(x_pos_bed_Q.shape[0], x_pos_bed_Q.shape[1], 1) + self.position(x_Q_1)

        index_I = np.arange(x_pos_bed_I.shape[1])
        random.shuffle(index_I)
        v_index_I = index_I[:-self.mask_len]
        m_index_I = index_I[-self.mask_len:]
        visible_I = x_pos_bed_I[:, v_index_I, :]
        rep_mask_token_I = rep_mask_token_I[:, m_index_I, :]

        index_Q = np.arange(x_pos_bed_Q.shape[1])
        random.shuffle(index_Q)
        v_index_Q = index_Q[:-self.mask_len]
        m_index_Q = index_Q[-self.mask_len:]
        visible_Q = x_pos_bed_Q[:, v_index_Q, :]
        rep_mask_token_Q = rep_mask_token_Q[:, m_index_Q, :]

        x_En_output_I = self.encoder_I(visible_I)
        x_En_output_Q = self.encoder_Q(visible_Q)

        x_En_share_input_IQ  = torch.cat([x_En_output_I, x_En_output_Q], dim=1)
        x_En_share_output_IQ = self.encoder_share(x_En_share_input_IQ)
        W1=int(x_En_share_output_IQ.shape[1]/2)
        x_En_share_output_I = x_En_share_output_IQ[:, 0 :W1,   :]
        x_En_share_output_Q = x_En_share_output_IQ[:, W1:2*W1, :]

#########################上面是Encoder相关部分，下面是Decoder相关部分########################

        x_decoder_input_I = torch.cat([x_En_share_output_I, rep_mask_token_I], dim=1)
        x_decoder_input_Q = torch.cat([x_En_share_output_Q, rep_mask_token_Q], dim=1)
        x_decoder_input_IQ = torch.cat([x_decoder_input_I, x_decoder_input_Q], dim=1)

        x_decoder_output_IQ = self.decoder(x_decoder_input_IQ)
        W2 = int(x_decoder_output_IQ.shape[1] / 2)
        x_decoder_output_I = x_decoder_output_IQ[:, 0 :W2,   :]
        x_decoder_output_Q = x_decoder_output_IQ[:, W2:W2*2, :]

        ########################################################################

        x_Norm_I = self.decoder_norm_I(x_decoder_output_I)      # 256,32,152
        x_Norm_Q = self.decoder_norm_Q(x_decoder_output_Q)      # 256,32,152
        x_Reconst_rep_I = self.decoder_pred_I(x_Norm_I)         # 256,32,152
        x_Reconst_rep_Q = self.decoder_pred_Q(x_Norm_Q)
        pred_I = self.output_proj_I_1(x_Reconst_rep_I.transpose(1, 2)).transpose(1, 2).contiguous() # bs,32,2
        pred_Q = self.output_proj_Q_1(x_Reconst_rep_Q.transpose(1, 2)).transpose(1, 2).contiguous() # bs,32,2

        pred_output_I = self.output_proj_I_2(pred_I)                                              # bs,128,2
        pred_output_Q = self.output_proj_Q_2(pred_Q)

        return (x_pos_bed_I, x_Reconst_rep_I,
                x_pos_bed_Q, x_Reconst_rep_Q,
                x_I_exp, pred_output_I,
                x_Q_exp, pred_output_Q,
                m_index_I, m_index_Q)

    def forward(self, x):
        if self.linear_proba:
            with torch.no_grad():
                x_I = x[:, :, 0]
                x_Q = x[:, :, 1]
                x_I_exp = x_I.unsqueeze(dim=2)  # 256,1,128
                x_Q_exp = x_Q.unsqueeze(dim=2)  # 256,1,128
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

                return x_En_share_output_IQ_mean

        else:
            x_I = x[:, :, 0]
            x_Q = x[:, :, 1]
            x_I_exp = x_I.unsqueeze(dim=2)  # 256,128,1
            x_Q_exp = x_Q.unsqueeze(dim=2)  # 256,128,1
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

            x_En_share_output_IQ_mean = torch.mean(x_En_share_output_IQ, dim=1) # 256,152

            x_MLP_1 = self.predict_head_1(x_En_share_output_IQ_mean)
            x_MLP_2 = self.predict_head_2(x_MLP_1)
            x_MLP_3 = self.predict_head_3(x_MLP_2)
            return x_MLP_3

