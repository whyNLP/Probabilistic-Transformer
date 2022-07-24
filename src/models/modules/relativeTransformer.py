import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import os
DEBUG=os.environ.get('DEBUG')
DRAW=os.environ.get('DRAW')

if DRAW: from .recorder import HeadRecorder

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=256, n_head=4, d_qkv=32, dropout=0.1, e_length=5, mode="k", **kwargs):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_qkv = d_qkv
        self.dropout = dropout

        # Positional embedding
        self.e_length = e_length # relative positional embedding length
        self.mode = mode # combination of 'q', 'k', 'v'. e.g. 'qk'
        if 'q' in mode:
            self.k_embedding = nn.Embedding(2 * e_length - 1, d_qkv)
        if 'k' in mode:
            self.q_embedding = nn.Embedding(2 * e_length - 1, d_qkv)
        if 'v' in mode:
            self.v_embedding = nn.Embedding(2 * e_length - 1, d_qkv)

        # We provide these model parameters to give an example of a weight
        # initialization approach that we know works well for our tasks. Feel free
        # to delete these lines and write your own, if you would like.
        self.w_q = nn.Parameter(torch.Tensor(n_head, d_model, d_qkv))
        self.w_k = nn.Parameter(torch.Tensor(n_head, d_model, d_qkv))
        self.w_v = nn.Parameter(torch.Tensor(n_head, d_model, d_qkv))
        self.w_o = nn.Parameter(torch.Tensor(n_head, d_qkv, d_model))
        nn.init.xavier_normal_(self.w_q)
        nn.init.xavier_normal_(self.w_k)
        nn.init.xavier_normal_(self.w_v)
        nn.init.xavier_normal_(self.w_o)

        # The hyperparameters given as arguments to this function and others
        # should be sufficient to reach the score targets for this project

        """YOUR CODE HERE"""
        self.dropout_s = nn.Dropout(p = dropout)
        self.dropout_o = nn.Dropout(p = dropout)
    
    def forward(self, x, mask):
        """Runs the multi-head self-attention layer.

        Args:
          x: the input to the layer, a tensor of shape [batch size, length, d_model]
          mask: a mask for disallowing attention to padding tokens. You will need to
                construct the mask yourself further on in this notebook. You may
                implement masking in any way; there is no requirement that you use
                a particular form for the mask object.
                a tensor of shape [batch size, length]
        Returns:
          A single tensor containing the output from this layer
            [batch size, length, d_model]
        """
        """YOUR CODE HERE"""
        # Implementation tip: using torch.einsum will greatly simplify the code that
        # you need to write.

        q = torch.einsum('blm,nmh->bnlh', [x, self.w_q]) # (batch size, n_head, length, d_qkv)
        k = torch.einsum('blm,nmh->bnlh', [x, self.w_k]) # (batch size, n_head, length, d_qkv)
        v = torch.einsum('blm,nmh->bnlh', [x, self.w_v]) # (batch size, n_head, length, d_qkv)

        length = x.shape[1]
        position_ids_l = torch.arange(length, dtype=torch.long, device=x.device).view(-1, 1)
        position_ids_r = torch.arange(length, dtype=torch.long, device=x.device).view(1, -1)
        distance = position_ids_l - position_ids_r
        distance[distance >= self.e_length] = self.e_length - 1
        distance[distance <= -self.e_length] = 1 - self.e_length

        scores = torch.matmul(q, k.transpose(-2, -1))
        
        if 'q' in self.mode:
            k_embed = self.k_embedding(distance + self.e_length - 1) # i, j, d_qkv
            scores_k = torch.einsum("bnjd,ijd->bnij", [k, k_embed])
            scores = scores + scores_k
        if 'k' in self.mode:
            q_embed = self.q_embedding(distance + self.e_length - 1)
            scores_q = torch.einsum("bnid,ijd->bnij", [q, q_embed])
            scores = scores + scores_q

        scores = scores / math.sqrt(self.d_qkv) # (batch size, n_head, length, length)
        if mask is not None:
            scores = scores.masked_fill((mask == 0).unsqueeze(-2).unsqueeze(1), -1e9)
        p_attn = F.softmax(scores, dim = -1)

        if DRAW: self.p_attn = p_attn
        
        p_attn = self.dropout_s(p_attn)
        
        x = torch.matmul(p_attn, v) # (batch size, n_head, length, d_qkv)

        if 'v' in self.mode:
            v_embed = self.v_embedding(distance + self.e_length - 1)
            v_x = torch.einsum('bnij,ijd->bnid', [p_attn, v_embed])
            x = x + v_x

        x = torch.matmul(x, self.w_o) # (batch size, n_head, length, d_model)
        return self.dropout_o(x.sum(dim = 1)) # (batch size, length, d_model)

    def _get_hyperparams(self):
        model_hps = {
            "d_model": self.d_model,
            "n_head": self.n_head,
            "d_qkv": self.d_qkv,
            "dropout": self.dropout,
            "e_length": self.e_length,
            "mode": self.mode
        }
        return model_hps
    
    def extra_repr(self) -> str:
        return ", ".join(["{}={}".format(k,v) for k,v in self._get_hyperparams().items()])

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff=1024, dropout=0.1):
        super().__init__()
        """YOUR CODE HERE"""

        self.d_model = d_model
        self.d_ff = d_ff

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p = dropout)

    def forward(self, x, *args):
        """YOUR CODE HERE"""

        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, d_model=256, d_ff=1024, n_head=4, d_qkv=32,
                dropout=0.1, e_length=5, mode='k'):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, n_head, d_qkv, dropout, e_length, mode)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(2)])
        self.d_model = d_model

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.layer_norm[0](x + self.self_attn(x, mask))
        return self.layer_norm[1](x + self.feed_forward(x))

class AddPositionalEncoding(nn.Module):
  def __init__(self, d_model=256, input_dropout=0.1, timing_dropout=0.1,
               max_len=512):
    super().__init__()
    self.timing_table = nn.Parameter(torch.FloatTensor(max_len, d_model))
    nn.init.normal_(self.timing_table)
    self.input_dropout = nn.Dropout(input_dropout)
    self.timing_dropout = nn.Dropout(timing_dropout)
  
  def forward(self, x):
    """
    Args:
      x: A tensor of shape [batch size, length, d_model]
    """
    x = self.input_dropout(x)
    timing = self.timing_table[None, :x.shape[1], :]
    timing = self.timing_dropout(timing)
    return x + timing

class COSPositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model=512, dropout=0.1, max_len=5000):
        super(COSPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + torch.autograd.Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)

class RelativeTransformerEncoder(nn.Module):
    def __init__(self, d_model=256, d_ff=1024, n_layers=4, n_head=4, d_qkv=32,
                dropout=0.1, pos_embed='none', e_length=5, mode='k'):
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
        self.e_length = e_length
        self.mode = mode

        if pos_embed == 'none':
            self.add_timing = lambda x: x
        elif pos_embed == 'add':
            self.add_timing = AddPositionalEncoding(d_model=d_model)
        elif pos_embed == 'cos':
            self.add_timing = COSPositionalEncoding(d_model=d_model, max_len=512)

        self.sublayers = nn.ModuleList([EncoderLayer(d_model, d_ff, n_head, d_qkv, dropout, e_length, mode) for _ in range(n_layers)])

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

        if DRAW:

            self.recorder = HeadRecorder(use_root=False)
            for i, sublayer in enumerate(self.sublayers):
                x = sublayer(x, mask)
                self.recorder[i] = sublayer.self_attn.p_attn
        
        else:

            for sublayer in self.sublayers:
                x = sublayer(x, mask)
        
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
            "e_length": self.e_length,
            "mode": self.mode
        }
        return model_hps

    def extra_repr(self) -> str:
        return ", ".join(["{}={}".format(k,v) for k,v in self._get_hyperparams().items()])