import torch
import torch.nn as nn
import torch.nn.functional as F

import opt_einsum as oe
import math

import os
DEBUG=os.environ.get('DEBUG')

class ProbEncoder(nn.Module):
    """
    Basic Probabilistic Transformer encoder.
    """
    def __init__(self, 
            d_model: int = 32, 
            d_label: int = 32, 
            n_iter: int = 4, 
            zero_edge: bool = False, 
            damping: float = 0, 
            stepsize: float = 1, 
            regularize: float = 1,
            norm: str = 'softmax', 
            dists: str = "",
            async_update: bool = True,
            output_prob: bool = False,
            use_td: str = 'no',
            dropout: float = 0.1
        ):
        """
        Initialize a basic Probabilistic Transformer encoder.
        :param d_model: dimensions of Z nodes.
        :param d_label: dimensions of Y nodes.
        :param n_iter: number of iterations.
        :param zero_edge: whether enforce zero edge between Z nodes.
        :param damping: damping of update. 0 means no damping is applied.
        :param stepsize: step size of update. 1 means full update is applied.
        :param regularize: regularization for MFVI. See 'Regularized Frank-Wolfe for Dense CRFs: 
                           GeneralizingMean Field and Beyond' (Ð.Khuê Lê-Huu, 2021) for details.
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
        """
        super().__init__()
        self.d_model = d_model
        self.d_label = d_label
        self.n_iter = n_iter
        self.zero_edge = zero_edge
        self.damping = damping
        self.stepsize = stepsize
        self.regularize = regularize
        self.norm = norm
        self.dists = dists
        self._dists = sorted([int(n) for n in dists.replace(' ', '').split(',') if n])
        self.async_update = async_update
        self.output_prob = output_prob
        self.use_td = use_td.replace(' ', '')
        self.dropout = dropout

        assert not self._dists or all([n>1 for n in self._dists]), "The minimum seperate point should be 2. See docs about distance."

        if self.norm == 'softmax':
            self.norm_func = lambda x: F.softmax(x, dim=-1)
        elif self.norm == 'relu':
            self.norm_func = lambda x: F.normalize(F.relu(x), p=1, dim=-1)
        else:
            raise ValueError("%s is not a normalization method." % self.norm)

        if self.use_td.startswith('uv:'):
            rank = int(self.use_td.split(':')[-1])
            self.U = nn.Parameter(torch.ones(2, len(self._dists)+1, self.d_model, rank, self.d_label))
            self.V = nn.Parameter(torch.ones(2, len(self._dists)+1, self.d_model, rank, self.d_label))
            nn.init.normal_(self.U, std = 1/math.sqrt(rank))
            nn.init.normal_(self.V, std = 1/math.sqrt(rank))
        elif self.use_td.startswith('uvw:'):
            rank = int(self.use_td.split(':')[-1])
            self.U = nn.Parameter(torch.Tensor(2, len(self._dists)+1, self.d_model, rank))
            self.V = nn.Parameter(torch.Tensor(2, len(self._dists)+1, self.d_model, rank))
            self.W = nn.Parameter(torch.Tensor(2, len(self._dists)+1, self.d_label, rank))
            nn.init.normal_(self.U, std = 1/math.sqrt(rank))
            nn.init.normal_(self.V, std = 1/math.sqrt(rank))
            nn.init.normal_(self.W, std = 1/math.sqrt(rank))
        else:
            self.ternary = nn.Parameter(torch.ones(2, len(self._dists)+1, self.d_model, self.d_model, self.d_label))
            nn.init.normal_(self.ternary)
        
        self.dropout_y = nn.Dropout(p = dropout)
        self.dropout_z = nn.Dropout(p = dropout)

    def forward(self, x, mask):
        batch_size, max_len, _ = x.shape

        ## Build mask
        mask1d = mask != 0
        mask2d = mask1d.unsqueeze(-1) * mask1d.unsqueeze(-2)
        mask1d = ~mask1d.view(batch_size, max_len, 1)
        mask2d = ~mask2d.view(batch_size, max_len, max_len, 1)

        distmask = torch.ones(len(self._dists)+1, max_len, max_len, dtype=torch.float).triu(1).to(x.device)
        if len(self._dists) > 0: # At least two dist blocks
            distmask[0] = distmask[0].tril(self._dists[0]-1)
            for i in range(1, len(self._dists)):
                ni_1, ni = self._dists[i-1], self._dists[i]
                distmask[i] = distmask[i].triu(ni_1) - distmask[i].triu(ni)
            distmask[-1] = distmask[-1].triu(self._dists[-1])

        ## Unary score
        unary = x # (batch_size, max_len, d_model)

        ## Recover ternary score
        if self.use_td.startswith('uv:'):
            ternary = oe.contract('tkadc,tkbdc->tkabc', *[self.U, self.V], backend='torch')
        elif self.use_td.startswith('uvw:'):
            ternary = oe.contract('tkad,tkbd,tkcd->tkabc', *[self.U, self.V, self.W], backend='torch')
        else:
            ternary = self.ternary

        ## Zero edge
        if self.zero_edge:
            ternary[..., 0] = 0

        ## Init with unary score
        q_z = unary.clone()
        q_y = torch.ones(batch_size, max_len, max_len, self.d_label).to(x.device)

        ## Initialization for async Y nodes
        cache_qy = q_y.clone()

        cache_norm_qz, cache_norm_qy = self.norm_func(q_z), self.norm_func(q_y)

        for _ in range(self.n_iter):

            if self.async_update:

                ## Update Y first
                cache_qz = q_z.clone()
                
                # Normalize
                q_z = (1-self.stepsize) * cache_norm_qz + self.stepsize * self.norm_func(q_z)
                cache_norm_qz = q_z.clone()
                
                # Apply mask
                q_z = q_z*(~mask1d) 
                
                # Calculate 2nd message for different dists
                second_order_message_G = oe.contract('zia,zjb,kabc,kij->zijc',*[q_z, q_z, self.d_model * ternary[0], distmask], backend='torch') + \
                                         oe.contract('zia,zjb,kbac,kji->zijc',*[q_z, q_z, self.d_model * ternary[1], distmask], backend='torch')
                
                # Update
                q_y = cache_qy * self.damping + second_order_message_G * (1-self.damping) / self.regularize
                q_y = self.dropout_y(q_y)
            
                ## Then update Z
                cache_qy = q_y.clone()
                
                # Normalize
                q_y = (1-self.stepsize) * cache_norm_qy + self.stepsize * self.norm_func(q_y)
                cache_norm_qy = q_y.clone()
                
                # Apply mask
                q_y = q_y - torch.diag_embed(q_y.diagonal(dim1=1, dim2=2), dim1=1, dim2=2)
                q_y.masked_fill_(mask2d, 0)
                
                # Calculate 2nd message for different dists
                second_order_message_F = oe.contract('zjb,zijc,kabc,kij->zia', *[q_z, q_y, self.d_model * ternary[0], distmask], backend='torch') + \
                                         oe.contract('zjb,zjic,kabc,kij->zia', *[q_z, q_y, self.d_model * ternary[1], distmask], backend='torch') + \
                                         oe.contract('zjb,zjic,kbac,kji->zia', *[q_z, q_y, self.d_model * ternary[0], distmask], backend='torch') + \
                                         oe.contract('zjb,zijc,kbac,kji->zia', *[q_z, q_y, self.d_model * ternary[1], distmask], backend='torch')
                
                # Update
                q_z = cache_qz * self.damping + (unary + second_order_message_F) * (1-self.damping) / self.regularize / self.d_model
                q_z = self.dropout_z(q_z)

            else:

                cache_qz, cache_qy = q_z.clone(), q_y.clone()
                
                # Normalize
                q_z, q_y = (1-self.stepsize) * cache_norm_qz + self.stepsize * self.norm_func(q_z), \
                           (1-self.stepsize) * cache_norm_qy + self.stepsize * self.norm_func(q_y)
                
                cache_norm_qz, cache_norm_qy = q_z.clone(), q_y.clone()
                
                # Apply mask
                q_z = q_z*(~mask1d) 
                q_y = q_y - torch.diag_embed(q_y.diagonal(dim1=1, dim2=2), dim1=1, dim2=2)
                q_y.masked_fill_(mask2d, 0)
                
                # Calculate 2nd message for different dists
                second_order_message_F = oe.contract('zjb,zijc,kabc,kij->zia', *[q_z, q_y, self.d_model * ternary[0], distmask], backend='torch') + \
                                         oe.contract('zjb,zjic,kabc,kij->zia', *[q_z, q_y, self.d_model * ternary[1], distmask], backend='torch') + \
                                         oe.contract('zjb,zjic,kbac,kji->zia', *[q_z, q_y, self.d_model * ternary[0], distmask], backend='torch') + \
                                         oe.contract('zjb,zijc,kbac,kji->zia', *[q_z, q_y, self.d_model * ternary[1], distmask], backend='torch')

                second_order_message_G = oe.contract('zia,zjb,kabc,kij->zijc',*[q_z, q_z, self.d_model * ternary[0], distmask], backend='torch') + \
                                         oe.contract('zia,zjb,kbac,kji->zijc',*[q_z, q_z, self.d_model * ternary[1], distmask], backend='torch')
                
                # Update
                q_y = cache_qy * self.damping + second_order_message_G * (1-self.damping) / self.regularize
                q_z = cache_qz * self.damping + (unary + second_order_message_F) * (1-self.damping) / self.regularize / self.d_model
                q_y = self.dropout_y(q_y)
                q_z = self.dropout_z(q_z)
                
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
                print("Q_z:")
                print(q_z)
                print(torch.mean(q_z))
                print(torch.std(q_z))
                print("Q_z norm:")
                print(self.norm_func(q_z))
                print(torch.mean(self.norm_func(q_z)))
                print("second_order_message_G:")
                print(second_order_message_G)
                print(torch.mean(second_order_message_G))
                print(torch.std(second_order_message_G))
                print("Q_y:")
                print(q_y)
                print(torch.mean(q_y))
                print(torch.std(q_y))
                print("Q_y norm:")
                print(self.norm_func(q_y))
                print(torch.mean(self.norm_func(q_y)))
                print("ternary:")
                print(ternary)
                print(torch.mean(ternary))
                print(torch.std(ternary))
                exit(0)
            
            self.cnt += 1
        
        # Save the Q value for Y nodes
        if not self.async_update:
            q_y = (1-self.stepsize) * cache_norm_qy + self.stepsize * self.norm_func(q_y)
            q_y = q_y - torch.diag_embed(q_y.diagonal(dim1=1, dim2=2), dim1=1, dim2=2)
            q_y.masked_fill_(mask2d, 0)
        self.q_y = q_y

        if self.output_prob:
            q_z = (1-self.stepsize) * cache_norm_qz + self.stepsize * self.norm_func(q_z)
            q_z = q_z*(~mask1d)
        return q_z
    
    def getTernaryNorm(self, p):
        ## Recover ternary score
        if self.use_td.startswith('uv:'):
            ternary = oe.contract('tkadc,tkbdc->tkabc', *[self.U, self.V], backend='torch')
        elif self.use_td.startswith('uvw:'):
            ternary = oe.contract('tkad,tkbd,tkcd->tkabc', *[self.U, self.V, self.W], backend='torch')
        else:
            ternary = self.ternary
        ## Zero edge
        if self.zero_edge:
            ternary[..., 0] = 0
        
        return ternary.norm(p=p)
    
    def getYNorm(self, p):
        return (1-self.q_y[..., 0])[self.q_y[..., 0] > 0].norm(p=p)
    
    def getYEntropy(self):
        nums = (self.q_y[..., 0] > 0).count_nonzero()
        entropy = -1 * torch.sum(self.q_y * (self.q_y + 1e-7).log()) / nums
        return entropy

    def _get_hyperparams(self):
        model_hps = {
            "d_model": self.d_model,
            "d_label": self.d_label,
            "n_iter": self.n_iter,
            "zero_edge": self.zero_edge,
            "damping": self.damping,
            "stepsize": self.stepsize,
            "regularize": self.regularize,
            "norm": self.norm,
            "dists": self.dists,
            "async_update": self.async_update,
            "output_prob": self.output_prob,
            'use_td': self.use_td,
            "dropout": self.dropout
        }
        return model_hps
    
    def extra_repr(self) -> str:
        return ", ".join(["{}={}".format(k,v) for k,v in self._get_hyperparams().items()])
