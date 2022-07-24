import torch
import torch.nn as nn
import torch.nn.functional as F

import opt_einsum as oe
import math

from .transformer import AddPositionalEncoding, COSPositionalEncoding

import os
DEBUG=os.environ.get('DEBUG')
DRAW=os.environ.get('DRAW')

if DRAW: from .recorder import HeadRecorder

class AbsoluteHeadProbEncoder(nn.Module):
    """
    Head Probabilistic Transformer encoder.
    """
    def __init__(self, 
            d_model: int = 32, 
            n_head: int = 10, 
            n_iter: int = 4, 
            damping_H: float = 0, 
            damping_Z: float = 0, 
            stepsize_H: float = 1, 
            stepsize_Z: float = 1, 
            regularize_H: float = 1,
            regularize_Z: float = 1,
            norm: str = 'softmax', 
            pos_embed: str = "add",
            async_update: bool = True,
            output_prob: bool = False,
            use_td: str = 'no',
            dropout: float = 0,
            block_msg: bool = False
        ):
        """
        Initialize a basic Probabilistic Transformer encoder.
        :param d_model: dimensions of Z nodes.
        :param n_head: number of heads.
        :param n_iter: number of iterations.
        :param damping_H: damping of H nodes update. 0 means no damping is applied.
        :param damping_Z: damping of Z nodes update. 0 means no damping is applied.
        :param stepsize_H: step size of H nodes update. 1 means full update is applied.
        :param stepsize_Z: step size of Z nodes update. 1 means full update is applied.
        :param regularize_H: regularization for updating H nodes.
        :param regularize_Z: regularization for updating Z nodes.
                'regularize_H' and 'regularize_Z' are regularizations for MFVI. See 
                'Regularized Frank-Wolfe for Dense CRFs: GeneralizingMean Field and 
                Beyond' (Ð.Khuê Lê-Huu, 2021) for details.
        :param norm: normalization method. Options: ['softmax', 'relu'], Default: 'softmax'.
        :param pos_embed: positional embedding. Options: ['none', 'add', 'cos'], Default: 'add'.
        :param async_update: update the q values asyncronously (Y first, then Z). Default: True.
        :param output_prob: If true, output a normalized probabilistic distribution. Otherwise
                            output unnormalized scores.
        :param use_td: control tensor decomposition. Options:
                         - 'no': no tensor decomposition;
                         - 'uv:{rank}': each 'head' decompose to 2 matrices W = U @ V. Use a
                           number to set the rank, e.g. 'uv:64';
                         - 'uvw:{rank}': decompose to sum of product of 3 vectors 
                           W = \sum U * V * W, where * is the outer product. Use a number to set
                           the rank, e.g. 'uvw:64'.
                       Default: 'no'.
        :param dropout: dropout for training. Default: 0.1.
        :param block_msg: block the message passed to Z_j in factor (H_i=k, Z_i=a, Z_j=b). Default: False.
        """
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.n_iter = n_iter
        self.damping_H = damping_H
        self.damping_Z = damping_Z
        self.stepsize_H = stepsize_H
        self.stepsize_Z = stepsize_Z
        self.regularize_H = regularize_H
        self.regularize_Z = regularize_Z
        self.norm = norm
        self.pos_embed = pos_embed
        self.async_update = async_update
        self.output_prob = output_prob
        self.use_td = use_td.replace(' ', '')
        self.dropout = dropout
        self.block_msg = block_msg

        if pos_embed == 'none':
            self.add_timing = lambda x: x
        elif pos_embed == 'add':
            self.add_timing = AddPositionalEncoding(d_model=d_model)
        elif pos_embed == 'cos':
            self.add_timing = COSPositionalEncoding(d_model=d_model, max_len=512)
        else:
            raise ValueError(f"pos_embed should be one of 'none', 'add', 'cos', but find {pos_embed}")

        if self.norm == 'softmax':
            self.norm_func = lambda x: F.softmax(x, dim=-1)
        elif self.norm == 'relu':
            self.norm_func = lambda x: F.normalize(F.relu(x), p=1, dim=-1)
        else:
            raise ValueError("%s is not a normalization method." % self.norm)

        if self.use_td.startswith('uv:'):
            rank = int(self.use_td.split(':')[-1])
            self.U = nn.Parameter(torch.Tensor(self.d_model, rank, self.n_head))
            self.V = nn.Parameter(torch.Tensor(self.d_model, rank, self.n_head))
            nn.init.normal_(self.U, std = 1/math.sqrt(rank))
            nn.init.normal_(self.V, std = 1/math.sqrt(rank))
        elif self.use_td.startswith('uvw:'):
            rank = int(self.use_td.split(':')[-1])
            self.U = nn.Parameter(torch.Tensor(self.d_model, rank))
            self.V = nn.Parameter(torch.Tensor(self.d_model, rank))
            self.W = nn.Parameter(torch.Tensor(self.n_head, rank))
            nn.init.normal_(self.U, std = 1/math.sqrt(rank))
            nn.init.normal_(self.V, std = 1/math.sqrt(rank))
            nn.init.normal_(self.W, std = 1/math.sqrt(rank))
        else:
            self.ternary = nn.Parameter(torch.Tensor(self.d_model, self.d_model, self.n_head))
            nn.init.normal_(self.ternary)
        
        self.dropout_h = nn.Dropout(p = dropout)
        self.dropout_z = nn.Dropout(p = dropout)

    def forward(self, x, mask):
        batch_size, max_len, _ = x.shape

        ## Build mask
        mask1d = mask != 0
        mask2d = mask1d.unsqueeze(-1) * mask1d.unsqueeze(-2)
        mask1d = ~mask1d.view(batch_size, max_len, 1)
        mask2d = ~mask2d.view(batch_size, 1, max_len, max_len).repeat_interleave(self.n_head,1)

        x = self.add_timing(x)

        ## Unary score
        unary = x # (batch_size, max_len, d_model)

        ## Recover ternary score
        if self.use_td.startswith('uv:'):
            ternary = oe.contract('adc,bdc->abc', *[self.U, self.V], backend='torch')
        elif self.use_td.startswith('uvw:'):
            ternary = oe.contract('ad,bd,cd->abc', *[self.U, self.V, self.W], backend='torch')
        else:
            ternary = self.ternary

        ## Init with unary score
        q_z = unary.clone()
        q_h = torch.ones(batch_size, self.n_head, max_len, max_len).to(x.device)
                
        # Apply mask
        q_z = q_z*(~mask1d)
        q_h = q_h - torch.diag_embed(torch.ones_like(q_h[...,0])*(1e9), dim1=-1, dim2=-2)
        q_h.masked_fill_(mask2d, -1e9)

        ## Initialization for async Y nodes
        cache_qh = q_h.clone()

        cache_norm_qz, cache_norm_qh = self.norm_func(q_z), self.norm_func(q_h)

        for iteration in range(self.n_iter):

            if self.async_update:

                ## Update Y first
                cache_qz = q_z.clone()
                
                # Normalize
                q_z = ( (1-self.stepsize_Z) * cache_norm_qz + self.stepsize_Z * self.norm_func(q_z) ) if iteration else self.norm_func(q_z)
                cache_norm_qz = q_z.clone()
                
                # Apply mask
                q_z = q_z*(~mask1d) 
                
                # Calculate 2nd message for different dists
                second_order_message_F = oe.contract('zia,zjb,abc->zcij',*[q_z, q_z, ternary], backend='torch')
                
                # Update
                q_h = cache_qh * self.damping_H + second_order_message_F * (1-self.damping_H) / self.regularize_H * self.d_model

                # Apply mask
                q_h = q_h - torch.diag_embed(torch.ones_like(q_h[...,0])*(1e9), dim1=-1, dim2=-2)
                q_h.masked_fill_(mask2d, -1e9)
            
                ## Then update Z
                cache_qh = q_h.clone()
                
                # Normalize
                q_h = ( (1-self.stepsize_H) * cache_norm_qh + self.stepsize_H * self.norm_func(q_h) ) if iteration else self.norm_func(q_h)
                q_h = self.dropout_h(q_h)
                cache_norm_qh = q_h.clone()
                
                # Calculate 2nd message for different dists
                second_order_message_G = oe.contract('zjb,zcij,abc->zia', *[q_z, q_h, ternary], backend='torch')
                if not self.block_msg:
                    second_order_message_G = second_order_message_G + oe.contract('zjb,zcji,bac->zia', *[q_z, q_h, ternary], backend='torch')
                
                # Update
                q_z = cache_qz * self.damping_Z + (unary + second_order_message_G) * (1-self.damping_Z) / self.regularize_Z
                q_z = self.dropout_z(q_z)

            else:

                # Apply mask
                q_h = q_h - torch.diag_embed(torch.ones_like(q_h[...,0])*(1e9), dim1=-1, dim2=-2)
                q_h.masked_fill_(mask2d, -1e9)

                cache_qz, cache_qh = q_z.clone(), q_h.clone()
                
                # Normalize
                q_z, q_h = (1-self.stepsize_Z) * cache_norm_qz + self.stepsize_Z * self.norm_func(q_z), \
                           (1-self.stepsize_H) * cache_norm_qh + self.stepsize_H * self.norm_func(q_h)
                
                cache_norm_qz, cache_norm_qh = q_z.clone(), q_h.clone()
                
                # Apply mask
                q_z = q_z*(~mask1d) 
                
                # Calculate 2nd message for different dists
                second_order_message_F = oe.contract('zia,zjb,abc->zcij',*[q_z, q_z, ternary], backend='torch')
                
                second_order_message_G = oe.contract('zjb,zcij,abc->zia', *[q_z, q_h, ternary], backend='torch')
                if not self.block_msg:
                    second_order_message_G = second_order_message_G + oe.contract('zjb,zcji,bac->zia', *[q_z, q_h, ternary], backend='torch')
                
                # Update
                q_h = cache_qh * self.damping_H + second_order_message_F * (1-self.damping_H) / self.regularize_H * self.d_model
                q_z = cache_qz * self.damping_Z + (unary + second_order_message_G) * (1-self.damping_Z) / self.regularize_Z 
                q_h = self.dropout_h(q_h)
                q_z = self.dropout_z(q_z)
            
            if DRAW:
                if not iteration:
                    self.recorder = HeadRecorder(use_root=False)
                self.recorder[iteration] = q_h
                
        if DEBUG:
            if 'cnt' not in self.__dict__:
                self.cnt = 0
            if self.cnt % (int(DEBUG)) == (int(DEBUG)-1):
                print("unary:")
                print(unary)
                print(torch.mean(unary))
                print(torch.std(unary))
                print("second_order_message_F:")
                print(second_order_message_F)
                print(torch.mean(second_order_message_F))
                print(torch.std(second_order_message_F))
                print("second_order_message_G:")
                print(second_order_message_G)
                print(torch.mean(second_order_message_G))
                print(torch.std(second_order_message_G))
                print("Q_z:")
                print(q_z)
                print(torch.mean(q_z))
                print(torch.std(q_z))
                print("Q_z norm:")
                print(self.norm_func(q_z))
                print(torch.mean(self.norm_func(q_z)))
                print("Q_h:")
                print(q_h)
                print(torch.mean(q_h))
                print(torch.std(q_h))
                print("Q_h norm:")
                print(self.norm_func(q_h))
                print(torch.mean(self.norm_func(q_h)))
                print("ternary:")
                print(ternary)
                print(torch.mean(ternary))
                print(torch.std(ternary))
                exit(0)
            
            self.cnt += 1
        
        # Save the Q value for H nodes
        if not self.async_update:
            q_h = q_h - torch.diag_embed(torch.ones_like(q_h[...,0])*(1e9), dim1=-1, dim2=-2)
            q_h.masked_fill_(mask2d, -1e9)
            q_h = (1-self.stepsize_H) * cache_norm_qh + self.stepsize_H * self.norm_func(q_h)
        self.q_h = q_h

        if self.output_prob:
            q_z = (1-self.stepsize_Z) * cache_norm_qz + self.stepsize_Z * self.norm_func(q_z)
            q_z = q_z*(~mask1d)
        return q_z
    
    def getTernaryNorm(self, p):
        ## Recover ternary score
        if self.use_td.startswith('uv:'):
            ternary = oe.contract('adc,bdc->abc', *[self.U, self.V], backend='torch')
        elif self.use_td.startswith('uvw:'):
            ternary = oe.contract('ad,bd,cd->abc', *[self.U, self.V, self.W], backend='torch')
        else:
            ternary = self.ternary
        
        return ternary.norm(p=p)

    def _get_hyperparams(self):
        model_hps = {
            "d_model": self.d_model,
            "n_head": self.n_head,
            "n_iter": self.n_iter,
            "damping_H": self.damping_H,
            "damping_Z": self.damping_Z,
            "stepsize_H": self.stepsize_H,
            "stepsize_Z": self.stepsize_Z,
            "regularize_H": self.regularize_H,
            "regularize_Z": self.regularize_Z,
            "norm": self.norm,
            "pos_embed": self.pos_embed,
            "async_update": self.async_update,
            "output_prob": self.output_prob,
            'use_td': self.use_td,
            "dropout": self.dropout,
            "block_msg": self.block_msg
        }
        return model_hps
    
    def extra_repr(self) -> str:
        return ", ".join(["{}={}".format(k,v) for k,v in self._get_hyperparams().items()])
