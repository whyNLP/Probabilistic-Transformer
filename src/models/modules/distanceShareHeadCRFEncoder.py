import torch
import torch.nn as nn
import torch.nn.functional as F

import opt_einsum as oe
import math

import os
DEBUG=os.environ.get('DEBUG')
DRAW=os.environ.get('DRAW')

if DRAW: from .recorder import HeadRecorder


class DistanceShareInefficientHeadProbEncoder(nn.Module):
    """
    Head Probabilistic Transformer encoder.
    Share most parameters between distances. Similar to transformer RPE.
    Inefficient version for testing.
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
            dists: str = "",
            async_update: bool = True,
            output_prob: bool = False,
            use_td: str = 'no',
            dropout: float = 0,
            block_msg: bool = False,
            use_projection: bool = False
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
        :param dists: distance pattern. Each distance group will use different factors. 
                      Dists should be groups of numbers seperated by ','. Each number represents
                      a seperate point. Empty means all tenery factors share the same parameters.
                      Note that the minimum seperate point you input should be 2. Default: "".
                      E.g. "" -> [1, +oo)
                           "3" -> [1, 2), [3, +oo)
                           "2, 4" -> [1, 2), [2, 4), [4, +oo)
                                i.e. {1}, {2, 3}, [4, +oo)
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
        :param use_projection: project the output using a feedforward block like transformer. Default: False.
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
        self.dists = dists
        self._dists = sorted([int(n) for n in dists.replace(' ', '').split(',') if n])
        self.async_update = async_update
        self.output_prob = output_prob
        self.use_td = use_td.replace(' ', '')
        self.dropout = dropout
        self.block_msg = block_msg
        self.use_projection = use_projection

        assert not self._dists or all([n>1 for n in self._dists]), "The minimum seperate point should be 2. See docs about distance."

        if self.norm == 'softmax':
            self.norm_func = lambda x: F.softmax(x, dim=-1)
        elif self.norm == 'relu':
            self.norm_func = lambda x: F.normalize(F.relu(x), p=1, dim=-1)
        else:
            raise ValueError("%s is not a normalization method." % self.norm)

        if self.use_td.startswith('uv:'):
            rank = int(self.use_td.split(':')[-1])
            self.U1 = nn.Parameter(torch.Tensor(self.d_model-1, rank, self.n_head))
            self.V1 = nn.Parameter(torch.Tensor(self.d_model-1, rank, self.n_head))
            self.U2 = nn.Parameter(torch.Tensor(2*(len(self._dists)+1), rank))
            self.V2 = nn.Parameter(torch.Tensor(2*(len(self._dists)+1), rank))
            nn.init.normal_(self.U1, std = 1/math.sqrt(rank))
            nn.init.normal_(self.V1, std = 1/math.sqrt(rank))
            nn.init.normal_(self.U2, std = 1/math.sqrt(rank))
            nn.init.normal_(self.V2, std = 1/math.sqrt(rank))
        elif self.use_td.startswith('uvw:'):
            rank = int(self.use_td.split(':')[-1])
            self.U = nn.Parameter(torch.Tensor(2*(len(self._dists)+1), self.d_model, rank))
            self.V = nn.Parameter(torch.Tensor(2*(len(self._dists)+1), self.d_model, rank))
            self.W = nn.Parameter(torch.Tensor(2*(len(self._dists)+1), self.n_head, rank))
            nn.init.normal_(self.U, std = 1/math.sqrt(rank))
            nn.init.normal_(self.V, std = 1/math.sqrt(rank))
            nn.init.normal_(self.W, std = 1/math.sqrt(rank))
        else:
            self.ternary = nn.Parameter(torch.Tensor(2*(len(self._dists)+1), self.d_model, self.d_model, self.n_head))
            nn.init.normal_(self.ternary)
        
        self.dropout_h = nn.Dropout(p = dropout)
        self.dropout_z = nn.Dropout(p = dropout)

        ## Build dist mask in advance. This may accelerate the forward process.
        ## Pre-defined max length. If exceeded, calculate during the forward process.
        self.max_len = 150
        distmask = torch.ones(len(self._dists)+1, self.max_len, self.max_len, dtype=torch.float).triu(1)
        if len(self._dists) > 0: # At least two dist blocks
            distmask[0] = distmask[0].tril(self._dists[0]-1)
            for i in range(1, len(self._dists)):
                ni_1, ni = self._dists[i-1], self._dists[i]
                distmask[i] = distmask[i].triu(ni_1) - distmask[i].triu(ni)
            distmask[-1] = distmask[-1].triu(self._dists[-1])
        distmask = torch.cat((distmask, distmask.transpose(-2, -1)), dim=0)
        self.distmask = distmask

        if self.use_projection:
            self.mlp = nn.Sequential(
                nn.Linear(self.d_model, self.d_model*4),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.d_model*4, self.d_model),
            )

    def forward(self, x, mask):
        batch_size, max_len, _ = x.shape

        ## Build mask
        mask1d = mask != 0
        mask2d = mask1d.unsqueeze(-1) * mask1d.unsqueeze(-2)
        mask1d = ~mask1d.view(batch_size, max_len, 1)
        mask2d = ~mask2d.view(batch_size, 1, max_len, max_len).repeat_interleave(self.n_head,1)

        if self.max_len < max_len:
            distmask = torch.ones(len(self._dists)+1, max_len, max_len, dtype=torch.float).triu(1).to(x.device)
            if len(self._dists) > 0: # At least two dist blocks
                distmask[0] = distmask[0].tril(self._dists[0]-1)
                for i in range(1, len(self._dists)):
                    ni_1, ni = self._dists[i-1], self._dists[i]
                    distmask[i] = distmask[i].triu(ni_1) - distmask[i].triu(ni)
                distmask[-1] = distmask[-1].triu(self._dists[-1])
            distmask = torch.cat((distmask, distmask.transpose(-2, -1)), dim=0)
        else:
            distmask = self.distmask[:, :max_len, :max_len].to(x.device)

        ## Unary score
        unary = x # (batch_size, max_len, d_model)

        ## Recover ternary score
        if self.use_td.startswith('uv:'):
            ## Recover U and V
            U = torch.cat((
                self.U1.unsqueeze(0).repeat_interleave(2*(len(self._dists)+1), dim=0), 
                self.U2.unsqueeze(1).unsqueeze(-1).repeat_interleave(self.n_head, dim=-1)
            ), dim=1)
            V = torch.cat((
                self.V1.unsqueeze(0).repeat_interleave(2*(len(self._dists)+1), dim=0), 
                self.V2.unsqueeze(1).unsqueeze(-1).repeat_interleave(self.n_head, dim=-1)
            ), dim=1)

            expr_F = oe.contract_expression('zia,zjb,kadc,kbdc,kij->zcij',*[unary.shape, unary.shape, U, V, distmask], constants=[2,3,4], optimize='optimal')
            expr_G1 = oe.contract_expression('zjb,zcij,kadc,kbdc,kij->zia', *[unary.shape, (batch_size, self.n_head, max_len, max_len), U, V, distmask], constants=[2,3,4], optimize='optimal')
            expr_G2 = oe.contract_expression('zjb,zcji,kbdc,kadc,kji->zia', *[unary.shape, (batch_size, self.n_head, max_len, max_len), U, V, distmask], constants=[2,3,4], optimize='optimal')
        elif self.use_td.startswith('uvw:'):
            expr_F = oe.contract_expression('zia,zjb,kad,kbd,kcd,kij->zcij',*[unary.shape, unary.shape, self.U, self.V, self.W, distmask], constants=[2,3,4,5], optimize='optimal')
            expr_G1 = oe.contract_expression('zjb,zcij,kad,kbd,kcd,kij->zia', *[unary.shape, (batch_size, self.n_head, max_len, max_len), self.U, self.V, self.W, distmask], constants=[2,3,4,5], optimize='optimal')
            expr_G2 = oe.contract_expression('zjb,zcji,kbd,kad,kcd,kji->zia', *[unary.shape, (batch_size, self.n_head, max_len, max_len), self.U, self.V, self.W, distmask], constants=[2,3,4,5], optimize='optimal')
        else:
            expr_F = oe.contract_expression('zia,zjb,kabc,kij->zcij',*[unary.shape, unary.shape, self.ternary, distmask], constants=[2,3], optimize='optimal')
            expr_G1 = oe.contract_expression('zjb,zcij,kabc,kij->zia', *[unary.shape, (batch_size, self.n_head, max_len, max_len), self.ternary, distmask], constants=[2,3], optimize='optimal')
            expr_G2 = oe.contract_expression('zjb,zcji,kbac,kji->zia', *[unary.shape, (batch_size, self.n_head, max_len, max_len), self.ternary, distmask], constants=[2,3], optimize='optimal')

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
                second_order_message_F = expr_F(q_z, q_z, backend='torch')
                
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
                second_order_message_G = expr_G1(q_z, q_h, backend='torch')
                if not self.block_msg:
                    second_order_message_G = second_order_message_G + expr_G2(q_z, q_h, backend='torch')
                
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
                second_order_message_F = expr_F(q_z, q_z, backend='torch')
                second_order_message_G = expr_G1(q_z, q_h, backend='torch')
                if not self.block_msg:
                    second_order_message_G = second_order_message_G + expr_G2(q_z, q_h, backend='torch')
                
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
                # print("ternary:")
                # print(ternary)
                # print(torch.mean(ternary))
                # print(torch.std(ternary))
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

        if self.use_projection:
            q_z = q_z + self.mlp(q_z)
        return q_z
    
    def getTernaryNorm(self, p):
        ## Recover ternary score
        if self.use_td.startswith('uv:'):
            ternary = oe.contract('kadc,kbdc->kabc', *[self.U, self.V], backend='torch')
        elif self.use_td.startswith('uvw:'):
            ternary = oe.contract('kad,kbd,kcd->kabc', *[self.U, self.V, self.W], backend='torch')
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
            "dists": self.dists,
            "async_update": self.async_update,
            "output_prob": self.output_prob,
            'use_td': self.use_td,
            "dropout": self.dropout,
            "block_msg": self.block_msg,
            "use_projection": self.use_projection
        }
        return model_hps
    
    def extra_repr(self) -> str:
        return ", ".join(["{}={}".format(k,v) for k,v in self._get_hyperparams().items()])


    
class DistanceShareHeadProbEncoder(nn.Module):
    """
    Head Probabilistic Transformer encoder.
    Share most parameters between distances. Similar to transformer RPE.
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
            dists: str = "",
            async_update: bool = True,
            output_prob: bool = False,
            use_td: str = 'no',
            dropout: float = 0,
            block_msg: bool = False,
            use_projection: bool = False
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
        :param dists: distance pattern. Each distance group will use different factors. 
                      Dists should be groups of numbers seperated by ','. Each number represents
                      a seperate point. Empty means all tenery factors share the same parameters.
                      Note that the minimum seperate point you input should be 2. Default: "".
                      E.g. "" -> [1, +oo)
                           "3" -> [1, 2), [3, +oo)
                           "2, 4" -> [1, 2), [2, 4), [4, +oo)
                                i.e. {1}, {2, 3}, [4, +oo)
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
        :param use_projection: project the output using a feedforward block like transformer. Default: False.
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
        self.dists = dists
        self._dists = sorted([int(n) for n in dists.replace(' ', '').split(',') if n])
        self.async_update = async_update
        self.output_prob = output_prob
        self.use_td = use_td.replace(' ', '')
        self.dropout = dropout
        self.block_msg = block_msg
        self.use_projection = use_projection

        assert not self._dists or all([n>1 for n in self._dists]), "The minimum seperate point should be 2. See docs about distance."

        if self.norm == 'softmax':
            self.norm_func = lambda x: F.softmax(x, dim=-1)
        elif self.norm == 'relu':
            self.norm_func = lambda x: F.normalize(F.relu(x), p=1, dim=-1)
        else:
            raise ValueError("%s is not a normalization method." % self.norm)

        if self.use_td.startswith('uv:'):
            rank = int(self.use_td.split(':')[-1])
            self.U1 = nn.Parameter(torch.Tensor(self.d_model-1, rank, self.n_head))
            self.V1 = nn.Parameter(torch.Tensor(self.d_model-1, rank, self.n_head))
            self.U2 = nn.Parameter(torch.Tensor(2*(len(self._dists)+1), rank))
            self.V2 = nn.Parameter(torch.Tensor(2*(len(self._dists)+1), rank))
            nn.init.normal_(self.U1, std = 1/math.sqrt(rank))
            nn.init.normal_(self.V1, std = 1/math.sqrt(rank))
            nn.init.normal_(self.U2, std = 1/math.sqrt(rank))
            nn.init.normal_(self.V2, std = 1/math.sqrt(rank))
        elif self.use_td.startswith('uvw:'):
            rank = int(self.use_td.split(':')[-1])
            self.U = nn.Parameter(torch.Tensor(2*(len(self._dists)+1), self.d_model, rank))
            self.V = nn.Parameter(torch.Tensor(2*(len(self._dists)+1), self.d_model, rank))
            self.W = nn.Parameter(torch.Tensor(2*(len(self._dists)+1), self.n_head, rank))
            nn.init.normal_(self.U, std = 1/math.sqrt(rank))
            nn.init.normal_(self.V, std = 1/math.sqrt(rank))
            nn.init.normal_(self.W, std = 1/math.sqrt(rank))
        else:
            self.ternary = nn.Parameter(torch.Tensor(2*(len(self._dists)+1), self.d_model, self.d_model, self.n_head))
            nn.init.normal_(self.ternary)
        
        self.dropout_h = nn.Dropout(p = dropout)
        self.dropout_z = nn.Dropout(p = dropout)

        ## Build dist mask in advance. This may accelerate the forward process.
        ## Pre-defined max length. If exceeded, calculate during the forward process.
        self.max_len = 150
        distmask = torch.ones(len(self._dists)+1, self.max_len, self.max_len, dtype=torch.float).triu(1)
        if len(self._dists) > 0: # At least two dist blocks
            distmask[0] = distmask[0].tril(self._dists[0]-1)
            for i in range(1, len(self._dists)):
                ni_1, ni = self._dists[i-1], self._dists[i]
                distmask[i] = distmask[i].triu(ni_1) - distmask[i].triu(ni)
            distmask[-1] = distmask[-1].triu(self._dists[-1])
        distmask = torch.cat((distmask, distmask.transpose(-2, -1)), dim=0)
        self.distmask = distmask

        if self.use_projection:
            self.mlp = nn.Sequential(
                nn.Linear(self.d_model, self.d_model*4),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.d_model*4, self.d_model),
            )

    def forward(self, x, mask):
        batch_size, max_len, _ = x.shape

        ## Build mask
        mask1d = mask != 0
        mask2d = mask1d.unsqueeze(-1) * mask1d.unsqueeze(-2)
        mask1d = ~mask1d.view(batch_size, max_len, 1)
        mask2d = ~mask2d.view(batch_size, 1, max_len, max_len).repeat_interleave(self.n_head,1)

        if self.max_len < max_len:
            distmask = torch.ones(len(self._dists)+1, max_len, max_len, dtype=torch.float).triu(1).to(x.device)
            if len(self._dists) > 0: # At least two dist blocks
                distmask[0] = distmask[0].tril(self._dists[0]-1)
                for i in range(1, len(self._dists)):
                    ni_1, ni = self._dists[i-1], self._dists[i]
                    distmask[i] = distmask[i].triu(ni_1) - distmask[i].triu(ni)
                distmask[-1] = distmask[-1].triu(self._dists[-1])
            distmask = torch.cat((distmask, distmask.transpose(-2, -1)), dim=0)
        else:
            distmask = self.distmask[:, :max_len, :max_len].to(x.device)

        ## Unary score
        unary = x # (batch_size, max_len, d_model)

        ## Recover ternary score
        if self.use_td.startswith('uv:'):
            # rank = int(self.use_td.split(':')[-1])

            ## Recover U and V
            U = torch.cat((
                self.U1.unsqueeze(0).repeat_interleave(2*(len(self._dists)+1), dim=0), 
                self.U2.unsqueeze(1).unsqueeze(-1).repeat_interleave(self.n_head, dim=-1)
            ), dim=1)
            V = torch.cat((
                self.V1.unsqueeze(0).repeat_interleave(2*(len(self._dists)+1), dim=0), 
                self.V2.unsqueeze(1).unsqueeze(-1).repeat_interleave(self.n_head, dim=-1)
            ), dim=1)

            # expr_F = oe.contract_expression('zia,zjb,kadc,kbdc,kij->zcij',*[unary.shape, unary.shape, U, V, distmask], constants=[2,3,4], optimize='optimal')
            expr_G1 = oe.contract_expression('zjb,zcij,kadc,kbdc,kij->zia', *[unary.shape, (batch_size, self.n_head, max_len, max_len), U, V, distmask], constants=[2,3,4], optimize='optimal')
            expr_G2 = oe.contract_expression('zjb,zcji,kbdc,kadc,kji->zia', *[unary.shape, (batch_size, self.n_head, max_len, max_len), U, V, distmask], constants=[2,3,4], optimize='optimal')

            # Calculate expr_F
            # Only this accelerates the training?! why...
            def expr_F(x: torch.Tensor, *args, **kwargs):
                """
                :param x: tensor of shape [batch_size, seq_length, embedding_size]
                """
                x_most, x_last = x[...,:-1], x[...,-1]
                return expr_F.expr1(x_most, x_most) + expr_F.expr2(x_last, x_most) + expr_F.expr3(x_most, x_last) + expr_F.expr4(x_last, x_last).unsqueeze(1)
            expr_F.expr1 = oe.contract_expression('zia,zjb,adc,bdc,kij->zcij', *[(batch_size, max_len, self.d_model-1), (batch_size, max_len, self.d_model-1), self.U1, self.V1, distmask], constants=[2,3,4], optimize='optimal')
            expr_F.expr2 = oe.contract_expression('zi,kd,bdc,zjb,kij->zcij', *[(batch_size, max_len), self.U2, self.V1, (batch_size, max_len, self.d_model-1), distmask], constants=[1,2,4], optimize='optimal')
            expr_F.expr3 = oe.contract_expression('zia,adc,kd,zj,kij->zcij', *[(batch_size, max_len, self.d_model-1), self.U1, self.V2, (batch_size, max_len), distmask], constants=[1,2,4], optimize='optimal')
            expr_F.expr4 = oe.contract_expression('zi,kd,kd,zj,kij->zij', *[(batch_size, max_len), self.U2, self.V2, (batch_size, max_len), distmask], constants=[1,2,4], optimize='optimal')

            # def expr_G1(x: torch.Tensor, h: torch.Tensor, **kwargs):
            #     """
            #     :param x: tensor of shape [batch_size, seq_length, embedding_size]
            #     :param h: tensor of shape [batch_size, num_heads, seq_length, seq_length]
            #     """
            #     x_most, x_last = x[...,:-1], x[...,-1]

            #     v_most = torch.einsum('zjb,bdc->zcjd', [x_most, self.V1])
            #     expr1 = expr_G1.expr1(h, v_most)
            #     expr2 = expr_G1.expr2(h, x_last)
            #     expr12 = oe.contract('zicd,adc->zia', *[expr1 + expr2, self.U1])
            #     expr3 = expr_G1.expr3(h, v_most)
            #     expr4 = expr_G1.expr4(h, x_last)
            #     expr34 = (expr3 + expr4).unsqueeze(-1)

            #     return torch.cat((expr12, expr34), dim=-1)
            # expr_G1.expr1 = oe.contract_expression('zcij,zcjd,kij->zicd', *[(batch_size, self.n_head, max_len, max_len), (batch_size, self.n_head, max_len, rank), distmask], constants=[2], optimize='optimal')
            # expr_G1.expr2 = oe.contract_expression('zcij,zj,kd,kij->zicd', *[(batch_size, self.n_head, max_len, max_len), (batch_size, max_len), self.V2, distmask], constants=[2,3], optimize='optimal')
            # expr_G1.expr3 = oe.contract_expression('zcij,zcjd,kd,kij->zi', *[(batch_size, self.n_head, max_len, max_len), (batch_size, self.n_head, max_len, rank), self.U2, distmask], constants=[2,3], optimize='optimal')
            # expr_G1.expr4 = oe.contract_expression('zcij,zj,kd,kd,kij->zi', *[(batch_size, self.n_head, max_len, max_len), (batch_size, max_len), self.V2, self.U2, distmask], constants=[2,3,4], optimize='optimal')

            # def expr_G2(x: torch.Tensor, h: torch.Tensor, **kwargs):
            #     """
            #     :param x: tensor of shape [batch_size, seq_length, embedding_size]
            #     :param h: tensor of shape [batch_size, num_heads, seq_length, seq_length]
            #     """
            #     x_most, x_last = x[...,:-1], x[...,-1]

            #     o_most = torch.einsum('zia,adc->zcid', [x_most, self.U1])
            #     expr1 = expr_G2.expr1(h, o_most)
            #     expr2 = expr_G2.expr2(h, x_last)
            #     expr12 = oe.contract('zjcd,bdc->zjb', *[expr1 + expr2, self.V1])
            #     expr3 = expr_G2.expr3(h, o_most)
            #     expr4 = expr_G2.expr4(h, x_last)
            #     expr34 = (expr3 + expr4).unsqueeze(-1)

            #     return torch.cat((expr12, expr34), dim=-1)
            # expr_G2.expr1 = oe.contract_expression('zcij,zcid,kij->zjcd', *[(batch_size, self.n_head, max_len, max_len), (batch_size, self.n_head, max_len, rank), distmask], constants=[2], optimize='optimal')
            # expr_G2.expr2 = oe.contract_expression('zcij,zi,kd,kij->zjcd', *[(batch_size, self.n_head, max_len, max_len), (batch_size, max_len), self.U2, distmask], constants=[2,3], optimize='optimal')
            # expr_G2.expr3 = oe.contract_expression('zcij,zcid,kd,kij->zj', *[(batch_size, self.n_head, max_len, max_len), (batch_size, self.n_head, max_len, rank), self.V2, distmask], constants=[2,3], optimize='optimal')
            # expr_G2.expr4 = oe.contract_expression('zcij,zi,kd,kd,kij->zj', *[(batch_size, self.n_head, max_len, max_len), (batch_size, max_len), self.U2, self.V2, distmask], constants=[2,3,4], optimize='optimal')

            # Codes for testing
            # t = unary.softmax(dim=-1)
            # t = torch.randn_like(unary).to(device=x.device).softmax(dim=-1)
            # h = torch.randn(batch_size, self.n_head, max_len, max_len).to(device=x.device).softmax(dim=-1)
            # # diff = expr_F0(t) - expr_F(t)
            # diff = expr_G20(t, h) - expr_G2(t, h)
            # print(diff.norm(p=2))
            # exit()

        elif self.use_td.startswith('uvw:'):
            expr_F = oe.contract_expression('zia,zjb,kad,kbd,kcd,kij->zcij',*[unary.shape, unary.shape, self.U, self.V, self.W, distmask], constants=[2,3,4,5], optimize='optimal')
            expr_G1 = oe.contract_expression('zjb,zcij,kad,kbd,kcd,kij->zia', *[unary.shape, (batch_size, self.n_head, max_len, max_len), self.U, self.V, self.W, distmask], constants=[2,3,4,5], optimize='optimal')
            expr_G2 = oe.contract_expression('zjb,zcji,kbd,kad,kcd,kji->zia', *[unary.shape, (batch_size, self.n_head, max_len, max_len), self.U, self.V, self.W, distmask], constants=[2,3,4,5], optimize='optimal')
        else:
            expr_F = oe.contract_expression('zia,zjb,kabc,kij->zcij',*[unary.shape, unary.shape, self.ternary, distmask], constants=[2,3], optimize='optimal')
            expr_G1 = oe.contract_expression('zjb,zcij,kabc,kij->zia', *[unary.shape, (batch_size, self.n_head, max_len, max_len), self.ternary, distmask], constants=[2,3], optimize='optimal')
            expr_G2 = oe.contract_expression('zjb,zcji,kbac,kji->zia', *[unary.shape, (batch_size, self.n_head, max_len, max_len), self.ternary, distmask], constants=[2,3], optimize='optimal')

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
                second_order_message_F = expr_F(q_z, q_z, backend='torch')
                
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
                second_order_message_G = expr_G1(q_z, q_h, backend='torch')
                if not self.block_msg:
                    second_order_message_G = second_order_message_G + expr_G2(q_z, q_h, backend='torch')
                
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
                second_order_message_F = expr_F(q_z, q_z, backend='torch')
                second_order_message_G = expr_G1(q_z, q_h, backend='torch')
                if not self.block_msg:
                    second_order_message_G = second_order_message_G + expr_G2(q_z, q_h, backend='torch')
                
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
                # print("ternary:")
                # print(ternary)
                # print(torch.mean(ternary))
                # print(torch.std(ternary))
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

        if self.use_projection:
            q_z = q_z + self.mlp(q_z)
        return q_z
    
    def getTernaryNorm(self, p):
        ## Recover ternary score
        if self.use_td.startswith('uv:'):
            ternary = oe.contract('kadc,kbdc->kabc', *[self.U, self.V], backend='torch')
        elif self.use_td.startswith('uvw:'):
            ternary = oe.contract('kad,kbd,kcd->kabc', *[self.U, self.V, self.W], backend='torch')
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
            "dists": self.dists,
            "async_update": self.async_update,
            "output_prob": self.output_prob,
            'use_td': self.use_td,
            "dropout": self.dropout,
            "block_msg": self.block_msg,
            "use_projection": self.use_projection
        }
        return model_hps
    
    def extra_repr(self) -> str:
        return ", ".join(["{}={}".format(k,v) for k,v in self._get_hyperparams().items()])