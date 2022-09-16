import torch
import torch.nn as nn
import torch.nn.functional as F

import opt_einsum as oe
import math

import os
DEBUG=os.environ.get('DEBUG')
DRAW=os.environ.get('DRAW')

if DRAW: from .recorder import HeadRecorder

class DecomposedGlobalHeadProbEncoder(nn.Module):
    """
    Head Probabilistic Transformer encoder with global nodes.

    A combination of "SingleGlobalHeadProbEncoder" and "DecomposedHeadProbEncoder".
    """
    def __init__(self, 
            d_model: int = 32, 
            n_head: int = 10, 
            n_global: int = 64,
            n_iter: int = 4, 
            damping_H: float = 0, 
            damping_Z: float = 0, 
            stepsize_H: float = 1, 
            stepsize_Z: float = 1, 
            regularize_H: float = 1,
            regularize_Z: float = 1,
            norm: str = 'softmax', 
            distance_rank: int = 4,
            dists: int = 4,
            async_update: bool = True,
            output_prob: bool = False,
            use_td: str = 'no',
            dropout: float = 0,
            block_msg: bool = False,
            use_td_global: str = 'no',
            mode: str = 'single-channel'
        ):
        """
        Initialize a basic Probabilistic Transformer encoder.
        :param d_model: dimensions of Z nodes.
        :param n_head: number of heads.
        :param n_global: number of global nodes.
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
        :param distance_rank: Rank for decomposition in distance.
        :param dists: k for the max distance to distinguish.
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
        :param use_td_global: simulate the global node. This is the same as doing a tensor decomposition.
                         - 'no': no tensor decomposition;
                         - 'uv:{rank}': global score matrix decompose to 2 matrices W = U @ V. 
                           Use a number to set the rank (i.e. dimension of the global nodes),
                           e.g. 'uv:64';
                         - 'norm:{rank}': similar to the above but matrix U is normalized.
                           e.g. 'norm:64';
                       Default: 'no'.
        :param mode: mode for global nodes. Opitons: 'attn-split', 'single-channel', '[k]-channel'. Default: 'single-channel'.
        """
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.n_global = n_global
        self.n_iter = n_iter
        self.damping_H = damping_H
        self.damping_Z = damping_Z
        self.stepsize_H = stepsize_H
        self.stepsize_Z = stepsize_Z
        self.regularize_H = regularize_H
        self.regularize_Z = regularize_Z
        self.norm = norm
        self.distance_rank = distance_rank
        self.dists = dists
        self.async_update = async_update
        self.output_prob = output_prob
        self.use_td = use_td.replace(' ', '')
        self.dropout = dropout
        self.block_msg = block_msg
        self.use_td_global = use_td_global.replace(' ', '')
        self.mode = mode

        assert self.mode in ('attn-split', 'single-channel') or self.mode.endswith('-channel'), f"Unexpected mode: {self.mode}"

        if self.norm == 'softmax':
            self.norm_func = lambda x: F.softmax(x, dim=-1)
        elif self.norm == 'relu':
            self.norm_func = lambda x: F.normalize(F.relu(x), p=1, dim=-1)
        else:
            raise ValueError("%s is not a normalization method." % self.norm)

        k = self.distance_rank

        self.distribution_params = nn.Parameter(torch.Tensor(2*self.dists, k))
        nn.init.normal_(self.distribution_params)

        if self.use_td.startswith('uv:'):
            rank = int(self.use_td.split(':')[-1])
            self.U = nn.Parameter(torch.Tensor(k, self.d_model, rank, self.n_head))
            self.V = nn.Parameter(torch.Tensor(k, self.d_model, rank, self.n_head))
            nn.init.normal_(self.U, std = 1/math.sqrt(rank))
            nn.init.normal_(self.V, std = 1/math.sqrt(rank))
        elif self.use_td.startswith('uvw:'):
            rank = int(self.use_td.split(':')[-1])
            self.U = nn.Parameter(torch.Tensor(k, self.d_model, rank))
            self.V = nn.Parameter(torch.Tensor(k, self.d_model, rank))
            self.W = nn.Parameter(torch.Tensor(k, self.n_head, rank))
            nn.init.normal_(self.U, std = 1/math.sqrt(rank))
            nn.init.normal_(self.V, std = 1/math.sqrt(rank))
            nn.init.normal_(self.W, std = 1/math.sqrt(rank))
        else:
            self.ternary = nn.Parameter(torch.Tensor(k, self.d_model, self.d_model, self.n_head))
            nn.init.normal_(self.ternary)

        global_heads = 1 if self.mode == 'single-channel' else self.n_head
        if self.mode.endswith('-channel') and self.mode != 'single-channel':
            global_heads = int(self.mode.split('-')[0])

        if self.use_td_global.startswith('uv:'):
            rank = int(self.use_td_global.split(':')[-1])
            self.U_global = nn.Parameter(torch.Tensor(self.n_global, rank, global_heads))
            self.V_global = nn.Parameter(torch.Tensor(self.d_model, rank, global_heads))
            nn.init.normal_(self.U_global, std = 1/(math.sqrt(rank)*math.sqrt(self.d_model)))
            nn.init.normal_(self.V_global, std = 1/(math.sqrt(rank)*math.sqrt(self.d_model)))
        elif self.use_td_global.startswith('norm:'):
            rank = int(self.use_td_global.split(':')[-1])
            self.U_global = nn.Parameter(torch.Tensor(self.n_global, rank, global_heads))
            self.V_global = nn.Parameter(torch.Tensor(self.d_model, rank, global_heads))
            nn.init.normal_(self.U_global)
            nn.init.normal_(self.V_global, std = 1/math.sqrt(self.d_model))
        else:
            self.global_ = nn.Parameter(torch.Tensor(self.n_global, self.d_model, global_heads))
            nn.init.normal_(self.global_, std=1/math.sqrt(self.d_model))
        
        self.dropout_h = nn.Dropout(p = dropout)
        self.dropout_z = nn.Dropout(p = dropout)

    def forward(self, x, mask):
        batch_size, max_len, _ = x.shape

        ## Build mask
        mask1d = mask != 0
        mask2d = mask1d.unsqueeze(-1) * mask1d.unsqueeze(-2)
        mask1d = ~mask1d.view(batch_size, max_len, 1)
        mask2d = ~mask2d.view(batch_size, 1, max_len, max_len).repeat_interleave(self.n_head,1)
        # mask2d = torch.cat((mask2d, torch.zeros(batch_size, self.n_head, max_len, self.n_global).to(x.device, dtype=mask2d.dtype)), dim=-1)

        ## Build dist mask
        position_ids_l = torch.arange(max_len, device=x.device).view(-1, 1)
        position_ids_r = torch.arange(max_len, device=x.device).view(1, -1)
        distance = position_ids_l - position_ids_r
        distance[distance > self.dists] = self.dists
        distance[distance < -self.dists] = -self.dists

        params = torch.cat((torch.zeros(1, self.distance_rank, device=x.device), self.distribution_params), dim=0)
        distmask = params[distance].permute(2,0,1)

        ## Unary score
        unary = x # (batch_size, max_len, d_model)

        global_heads = 1 if self.mode == 'single-channel' else self.n_head
        if self.mode.endswith('-channel') and self.mode != 'single-channel':
            global_heads = int(self.mode.split('-')[0])

        ## Prepare q-value sizes
        size_qz = unary.shape
        size_qh1 = (batch_size, self.n_head, max_len, max_len)
        size_qh2 = (batch_size, global_heads, max_len, self.n_global)

        ## Recover ternary score
        if self.use_td.startswith('uv:'):
            expr_F = oe.contract_expression('zia,zjb,kadc,kbdc,kij->zcij',*[size_qz, size_qz, self.U, self.V, distmask], constants=[2,3,4], optimize='optimal')
            expr_G1 = oe.contract_expression('zjb,zcij,kadc,kbdc,kij->zia', *[size_qz, size_qh1, self.U, self.V, distmask], constants=[2,3,4], optimize='optimal')
            expr_G2 = oe.contract_expression('zjb,zcji,kbdc,kadc,kji->zia', *[size_qz, size_qh1, self.U, self.V, distmask], constants=[2,3,4], optimize='optimal')
        elif self.use_td.startswith('uvw:'):
            expr_F = oe.contract_expression('zia,zjb,kad,kbd,kcd,kij->zcij',*[size_qz, size_qz, self.U, self.V, self.W, distmask], constants=[2,3,4,5], optimize='optimal')
            expr_G1 = oe.contract_expression('zjb,zcij,kad,kbd,kcd,kij->zia', *[size_qz, size_qh1, self.U, self.V, self.W, distmask], constants=[2,3,4,5], optimize='optimal')
            expr_G2 = oe.contract_expression('zjb,zcji,kbd,kad,kcd,kji->zia', *[size_qz, size_qh1, self.U, self.V, self.W, distmask], constants=[2,3,4,5], optimize='optimal')
        else:
            expr_F = oe.contract_expression('zia,zjb,kabc,kij->zcij',*[size_qz, size_qz, self.ternary, distmask], constants=[2,3], optimize='optimal')
            expr_G1 = oe.contract_expression('zjb,zcij,kabc,kij->zia', *[size_qz, size_qh1, self.ternary, distmask], constants=[2,3], optimize='optimal')
            expr_G2 = oe.contract_expression('zjb,zcji,kbac,kji->zia', *[size_qz, size_qh1, self.ternary, distmask], constants=[2,3], optimize='optimal')

        ## Recover global score
        if self.use_td_global.startswith('uv:'):
            expr_global_F = oe.contract_expression('zia,jdc,adc->zcij',*[size_qz, self.U_global, self.V_global], constants=[1,2], optimize='optimal')
            expr_global_G = oe.contract_expression('zcij,jdc,adc->zia', *[size_qh2, self.U_global, self.V_global], constants=[1,2], optimize='optimal')
        elif self.use_td_global.startswith('norm:'):
            expr_global_F = oe.contract_expression('zia,jdc,adc->zcij',*[size_qz, self.U_global.softmax(dim=1), self.V_global], constants=[1,2], optimize='optimal')
            expr_global_G = oe.contract_expression('zcij,jdc,adc->zia', *[size_qh2, self.U_global.softmax(dim=1), self.V_global], constants=[1,2], optimize='optimal')
        else:
            expr_global_F = oe.contract_expression('zia,jac->zcij',*[size_qz, self.global_], constants=[1], optimize='optimal')
            expr_global_G = oe.contract_expression('zcij,jac->zia', *[size_qh2, self.global_], constants=[1], optimize='optimal')

        ## Init with unary score
        q_z = unary.clone()
        q_h1 = torch.ones(batch_size, self.n_head, max_len, max_len).to(x.device)
        q_h2 = torch.ones(batch_size, global_heads, max_len, self.n_global).to(x.device)
                
        # Apply mask
        q_z = q_z*(~mask1d)
        q_h1 = q_h1 - torch.diag_embed(torch.ones_like(q_h1[...,0])*(1e9), dim1=-1, dim2=-2)
        q_h1.masked_fill_(mask2d, -1e9)

        ## Initialization for async Y nodes
        cache_qh1, cache_qh2 = q_h1.clone(), q_h2.clone()

        cache_norm_qz, cache_norm_qh1, cache_norm_qh2 = self.norm_func(q_z), self.norm_func(q_h1), self.norm_func(q_h2)

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
                second_order_message_F1 = expr_F(q_z, q_z, backend='torch')
                second_order_message_F2 = expr_global_F(q_z, backend='torch')
                # second_order_message_F = torch.cat((second_order_message_F1, second_order_message_F2), dim=-1)
                
                # Update
                q_h1 = cache_qh1 * self.damping_H + second_order_message_F1 * (1-self.damping_H) / self.regularize_H * self.d_model
                q_h2 = cache_qh2 * self.damping_H + second_order_message_F2 * (1-self.damping_H) / self.regularize_H * self.d_model

                # Apply mask
                q_h1 = q_h1 - torch.diag_embed(torch.ones_like(q_h1[...,0])*(1e9), dim1=-1, dim2=-2)
                q_h1.masked_fill_(mask2d, -1e9)
            
                ## Then update Z
                cache_qh1, cache_qh2 = q_h1.clone(), q_h2.clone()
                
                # Normalize
                q_h1 = ( (1-self.stepsize_H) * cache_norm_qh1 + self.stepsize_H * self.norm_func(q_h1) ) if iteration else self.norm_func(q_h1)
                q_h1 = self.dropout_h(q_h1)
                cache_norm_qh1 = q_h1.clone()

                q_h2 = ( (1-self.stepsize_H) * cache_norm_qh2 + self.stepsize_H * self.norm_func(q_h2) ) if iteration else self.norm_func(q_h2)
                q_h2 = self.dropout_h(q_h2)
                cache_norm_qh2 = q_h2.clone()
                
                # Calculate 2nd message for different dists
                second_order_message_G = expr_G1(q_z, q_h1, backend='torch')
                if not self.block_msg:
                    second_order_message_G = second_order_message_G + expr_G2(q_z, q_h1, backend='torch')
                second_order_message_G = second_order_message_G + expr_global_G(q_h2, backend='torch')
                
                # Update
                q_z = cache_qz * self.damping_Z + (unary + second_order_message_G) * (1-self.damping_Z) / self.regularize_Z
                q_z = self.dropout_z(q_z)

            else:

                raise NotImplementedError

                # Apply mask
                q_h = q_h - torch.diag_embed(torch.ones_like(q_h[...,0])*(1e9), dim1=-1, dim2=-2)
                q_h.masked_fill_(mask2d, -1e9)

                cache_qz, cache_qh = q_z.clone(), q_h.clone()
                
                # Normalize
                q_z, q_h = (1-self.stepsize_Z) * cache_norm_qz + self.stepsize_Z * self.norm_func(q_z), \
                           (1-self.stepsize_H) * cache_norm_qh + self.stepsize_H * head_norm_func(q_h)
                
                cache_norm_qz, cache_norm_qh = q_z.clone(), q_h.clone()
                
                # Apply mask
                q_z = q_z*(~mask1d) 
                
                # Calculate 2nd message for different dists
                second_order_message_F = oe.contract('zia,zjb,kabc,kij->zcij',*[q_z, q_z, ternary, distmask], backend='torch')
                
                second_order_message_G = oe.contract('zjb,zcij,kabc,kij->zia', *[q_z, q_h, ternary, distmask], backend='torch')
                if not self.block_msg:
                    second_order_message_G = second_order_message_G + oe.contract('zjb,zcji,kbac,kji->zia', *[q_z, q_h, ternary, distmask], backend='torch')
                
                # Update
                q_h = cache_qh * self.damping_H + second_order_message_F * (1-self.damping_H) / self.regularize_H * self.d_model
                q_z = cache_qz * self.damping_Z + (unary + second_order_message_G) * (1-self.damping_Z) / self.regularize_Z 
                q_h = self.dropout_h(q_h)
                q_z = self.dropout_z(q_z)
            
            if DRAW:
                if not iteration:
                    self.recorder = HeadRecorder(use_root=False)
                self.recorder[iteration] = q_h1
                
        if DEBUG:
            if 'cnt' not in self.__dict__:
                self.cnt = 0
            if self.cnt % (int(DEBUG)) == (int(DEBUG)-1):
                print("unary:")
                print(unary)
                print(torch.mean(unary))
                print(torch.std(unary))
                print("second_order_message_F1:")
                print(second_order_message_F1)
                print(torch.mean(second_order_message_F1))
                print(torch.std(second_order_message_F1))
                print("second_order_message_F2:")
                print(second_order_message_F2)
                print(torch.mean(second_order_message_F2))
                print(torch.std(second_order_message_F2))
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
                # print("Q_h:")
                # print(q_h)
                # print(torch.mean(q_h))
                # print(torch.std(q_h))
                # print("Q_h norm:")
                # print(self.norm_func(q_h))
                # print(torch.mean(self.norm_func(q_h)))
                # print("ternary:")
                # print(ternary)
                # print(torch.mean(ternary))
                # print(torch.std(ternary))
                exit(0)
            
            self.cnt += 1
        
        # # Save the Q value for H nodes
        # if not self.async_update:
        #     q_h = (1-self.stepsize_H) * cache_norm_qh + self.stepsize_H * self.norm_func(q_h)
        #     q_h = q_h - torch.diag_embed(q_h.diagonal(dim1=1, dim2=2), dim1=1, dim2=2)
        #     q_h.masked_fill_(mask2d, 0)
        # self.q_h = q_h

        if self.output_prob:
            q_z = (1-self.stepsize_Z) * cache_norm_qz + self.stepsize_Z * self.norm_func(q_z)
            q_z = q_z*(~mask1d)
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

    def getGlobalNorm(self, p):
        ## Recover global score
        if self.use_td_global.startswith('uv:'):
            global_ = oe.contract('adc,bdc->abc', *[self.U_global, self.V_global], backend='torch')
        elif self.use_td_global.startswith('norm:'):
            global_ = oe.contract('adc,bdc->abc', *[self.U_global.softmax(dim=1), self.V_global], backend='torch')
        else:
            global_ = self.global_
        
        return global_.norm(p=p)

    def _get_hyperparams(self):
        model_hps = {
            "d_model": self.d_model,
            "n_head": self.n_head,
            "n_global": self.n_global,
            "n_iter": self.n_iter,
            "damping_H": self.damping_H,
            "damping_Z": self.damping_Z,
            "stepsize_H": self.stepsize_H,
            "stepsize_Z": self.stepsize_Z,
            "regularize_H": self.regularize_H,
            "regularize_Z": self.regularize_Z,
            "norm": self.norm,
            "distance_rank": self.distance_rank,
            "dists": self.dists,
            "async_update": self.async_update,
            "output_prob": self.output_prob,
            'use_td': self.use_td,
            "dropout": self.dropout,
            "block_msg": self.block_msg,
            "use_td_global": self.use_td_global,
            "mode": self.mode
        }
        return model_hps
    
    def extra_repr(self) -> str:
        return ", ".join(["{}={}".format(k,v) for k,v in self._get_hyperparams().items()])