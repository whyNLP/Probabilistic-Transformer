import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer import (
    MultiHeadAttention, 
    PositionwiseFeedForward,
    AddPositionalEncoding,
    COSPositionalEncoding,
    LCOSPositionalEncoding,
    EncoderLayer
)


class EmbedResidualEncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, d_model=256, d_ff=1024, n_head=4, d_qkv=32,
                dropout=0.1):
        super(EmbedResidualEncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, n_head, d_qkv, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(2)])
        self.d_model = d_model

    def forward(self, x, mask, x0):
        "Follow Figure 1 (left) for connections."
        x = self.layer_norm[0](x + self.self_attn(x, mask))
        return self.layer_norm[1](x + self.feed_forward(x))


class EmbedResidualEncoderLayerAdd(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, d_model=256, d_ff=1024, n_head=4, d_qkv=32,
                dropout=0.1):
        super(EmbedResidualEncoderLayerAdd, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, n_head, d_qkv, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(2)])
        self.d_model = d_model

    def forward(self, x, mask, x0):
        "Follow Figure 1 (left) for connections."
        x = self.layer_norm[0](x + x0 + self.self_attn(x, mask))
        return self.layer_norm[1](x + x0 + self.feed_forward(x))


class EmbedResidualEncoderLayerReplace(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, d_model=256, d_ff=1024, n_head=4, d_qkv=32,
                dropout=0.1):
        super(EmbedResidualEncoderLayerReplace, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, n_head, d_qkv, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(2)])
        self.d_model = d_model

    def forward(self, x, mask, x0):
        "Follow Figure 1 (left) for connections."
        x = self.layer_norm[0](x0 + self.self_attn(x, mask))
        return self.layer_norm[1](x0 + self.feed_forward(x))


class EmbedResidualEncoderLayerAverage(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, d_model=256, d_ff=1024, n_head=4, d_qkv=32,
                dropout=0.1):
        super(EmbedResidualEncoderLayerAverage, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, n_head, d_qkv, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(2)])
        self.d_model = d_model

    def forward(self, x, mask, x0):
        "Follow Figure 1 (left) for connections."
        x = self.layer_norm[0]((x + x0)*0.5 + self.self_attn(x, mask))
        return self.layer_norm[1]((x + x0)*0.5 + self.feed_forward(x))


class EmbedResidualEncoderLayerWeighted(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, d_model=256, d_ff=1024, n_head=4, d_qkv=32,
                dropout=0.1):
        super(EmbedResidualEncoderLayerWeighted, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, n_head, d_qkv, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(2)])
        self.d_model = d_model
        self.weight = nn.Parameter(torch.zeros(2))

    def forward(self, x, mask, x0):
        "Follow Figure 1 (left) for connections."
        p = torch.sigmoid(self.weight)
        x = self.layer_norm[0](x * p[0] + x0 * (1-p[0]) + self.self_attn(x, mask))
        return self.layer_norm[1](x * p[1] + x0 * (1-p[1]) + self.feed_forward(x))


class EmbedResidualEncoderLayerFreeWeighted(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, d_model=256, d_ff=1024, n_head=4, d_qkv=32,
                dropout=0.1):
        super(EmbedResidualEncoderLayerFreeWeighted, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, n_head, d_qkv, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(2)])
        self.d_model = d_model
        # self.weight = nn.Parameter(torch.tensor([1., 0., 1., 0.]))
        self.weight = nn.Parameter(torch.tensor([.5, .5, .5, .5]))

    def forward(self, x, mask, x0):
        "Follow Figure 1 (left) for connections."
        p = self.weight
        x = self.layer_norm[0](x * p[0] + x0 * p[1] + self.self_attn(x, mask))
        return self.layer_norm[1](x * p[2] + x0 * p[3] + self.feed_forward(x))


class EmbedResidualTransformerEncoder(nn.Module):
    def __init__(self, d_model=256, d_ff=1024, n_layers=4, n_head=4, d_qkv=32,
                dropout=0.1, pos_embed='none', mode='none'):
        super().__init__()
        # Implementation tip: if you are storing nn.Module objects in a list, use
        # nn.ModuleList. If you use assignment statements of the form
        # `self.sublayers = [x, y, z]` with a plain python list instead of a
        # ModuleList, you might find that none of the sub-layer parameters are
        # trained.

        """YOUR CODE HERE"""
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_layers = n_layers
        self.n_head = n_head
        self.d_qkv = d_qkv
        self.dropout = dropout
        self.pos_embed = pos_embed

        if pos_embed == 'none':
            self.add_timing = lambda x: x
        elif pos_embed == 'add':
            self.add_timing = AddPositionalEncoding(d_model=d_model)
        elif pos_embed == 'cos':
            self.add_timing = COSPositionalEncoding(d_model=d_model, max_len=512)
        elif pos_embed == 'lcos':
            self.add_timing = LCOSPositionalEncoding(d_model=d_model, max_len=512)

        if mode == 'none':
            layer = EmbedResidualEncoderLayer
        elif mode == 'add':
            layer = EmbedResidualEncoderLayerAdd
        elif mode == 'replace':
            layer = EmbedResidualEncoderLayerReplace
        elif mode == 'average':
            layer = EmbedResidualEncoderLayerAverage
        elif mode == 'weighted':
            layer = EmbedResidualEncoderLayerWeighted
        elif mode == 'free-weighted':
            layer = EmbedResidualEncoderLayerFreeWeighted
        elif mode == 'prior-weighted':
            self.sublayers = nn.ModuleList([EmbedResidualEncoderLayer(d_model, d_ff, n_head, d_qkv, dropout) for _ in range(n_layers-1)] + [EmbedResidualEncoderLayerReplace(d_model, d_ff, n_head, d_qkv, dropout)])
            return

        self.sublayers = nn.ModuleList([layer(d_model, d_ff, n_head, d_qkv, dropout) for _ in range(n_layers)])

    def forward(self, x, mask):
        """Runs the Transformer encoder.

        Args:
        x: the input to the Transformer, a tensor of shape
            [batch size, length, d_model]
        mask: a mask for disallowing attention to padding tokens. You will need to
                construct the mask yourself further on in this notebook. You may
                implement masking in any way; there is no requirement that you use
                a particular form for the mask object.
        Returns:
        A single tensor containing the output from the Transformer
            [batch size, length, d_model]
        """

        """YOUR CODE HERE"""
        x = self.add_timing(x)
        x0 = x.clone()

        for sublayer in self.sublayers:
            x = sublayer(x, mask, x0)
        return x

    def _get_hyperparams(self):
        model_hps = {
            "d_model": self.d_model,
            "d_ff": self.d_ff,
            "n_layers": self.n_layers,
            "n_head": self.n_head,
            "d_qkv": self.d_qkv,
            "dropout": self.dropout,
            "pos_embed": self.pos_embed,
        }
        return model_hps

    def extra_repr(self) -> str:
        return ", ".join(["{}={}".format(k,v) for k,v in self._get_hyperparams().items()])
