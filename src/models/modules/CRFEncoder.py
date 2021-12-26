import torch
import torch.nn as nn
import torch.nn.functional as F

import opt_einsum as oe

from deprecated import deprecated

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
            output_prob: bool = False,
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
                      Dists should be groups of numbers seperated by ','. Zero will be excluded.
                      Empty means all tenery factors share the same parameters. Default: "".
                      E.g. "1" -> (-oo, -1], [1, +oo)
                           "-3, -1, 2, 4" -> (-oo, -3), [-3, -1), [-1, 2), [2, 4), [4, +oo)
                                i.e. (-oo, -4], {-3, -2}, {-1, 1}, {2, 3}, [4, +oo)
                           "-2,-1,1,3" -> (-oo, -2), -2, -1, [1, 3), [3, +oo)
        :param output_prob: If true, output a normalized probabilistic distribution. Otherwise
                            output unnormalized scores.
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
        self.output_prob = output_prob

        if self.norm == 'softmax':
            self.norm_func = lambda x: F.softmax(x, dim=-1)
        elif self.norm == 'relu':
            self.norm_func = lambda x: F.normalize(F.relu(x), p=1, dim=-1)
        else:
            raise ValueError("%s is not a normalization method." % self.norm)

        self.ternary = nn.Parameter(torch.ones(2, len(self._dists)+1, self.d_model, self.d_model, self.d_label))
        ## Seems to have better performance
        nn.init.kaiming_normal_(self.ternary, nonlinearity='relu')

    def forward(self, x, mask):
        batch_size, max_len, _ = x.shape

        ## Build mask
        mask1d = mask != 0
        mask2d = mask1d.unsqueeze(-1) * mask1d.unsqueeze(-2)
        mask1d = ~mask1d.view(batch_size, max_len, 1)
        mask2d = ~mask2d.view(batch_size, max_len, max_len, 1)

        distmask = torch.ones(len(self._dists)+1, max_len, max_len, dtype=torch.float).to(x.device)
        if len(self._dists) > 0: # At least two dist blocks
            distmask[0] = distmask[0].tril(self._dists[0]-1)
            for i in range(1, len(self._dists)):
                ni_1, ni = self._dists[i-1], self._dists[i]
                distmask[i] = distmask[i].triu(ni_1) - distmask[i].triu(ni)
            distmask[-1] = distmask[-1].triu(self._dists[-1])

        ## Unary score
        unary = x # (batch_size, max_len, d_model)

        ## Zero edge
        if self.zero_edge:
            with torch.no_grad():
                self.ternary[..., 0] = 0

        ## Init with unary score
        q_z = unary.clone()
        q_y = torch.ones(batch_size, max_len, max_len, self.d_label).to(x.device)

        cache_norm_qz, cache_norm_qy = self.norm_func(q_z), self.norm_func(q_y)

        for _ in range(self.n_iter):
            cache_qz, cache_qy = q_z.clone(), q_y.clone()
            
            # Normalize
            q_z, q_y = self.stepsize * cache_norm_qz + (1-self.stepsize) * self.norm_func(q_z), \
                       self.stepsize * cache_norm_qy + (1-self.stepsize) * self.norm_func(q_y)
            
            cache_norm_qz, cache_norm_qy = q_z.clone(), q_y.clone()
            
            # Apply mask
            q_z = q_z*(~mask1d) 
            q_y = q_y - torch.diag_embed(q_y.diagonal(dim1=1, dim2=2), dim1=1, dim2=2)
            q_y.masked_fill_(mask2d, 0)
            
            # Calculate 2nd message for different dists
            second_order_message_F = oe.contract('zjb,zijc,kabc,kij->zia', *[q_z, q_y, self.ternary[0], distmask], backend='torch') + \
                                     oe.contract('zjb,zjic,kabc,kij->zia', *[q_z, q_y, self.ternary[1], distmask], backend='torch')

            second_order_message_G = oe.contract('zia,zjb,kabc,kij->zijc',*[q_z, q_z, self.ternary[0], distmask], backend='torch') + \
                                     oe.contract('zja,zib,kabc,kij->zijc',*[q_z, q_z, self.ternary[1], distmask], backend='torch')
            
            # Update
            q_y = cache_qy * self.damping + second_order_message_G * (1-self.damping) / self.regularize
            q_z = cache_qz * self.damping + (unary + second_order_message_F) * (1-self.damping) / self.regularize
                
        if DEBUG:
            if 'cnt' not in self.__dict__:
                self.cnt = 0
            if self.cnt % (2298*16-1) == 0:
                print("Q_z:")
                print(q_z)
                print(torch.mean(q_z))
                print("Q_y:")
                print(q_y)
                print(torch.mean(q_y))
                print("Q_y relu:")
                print(F.normalize(F.relu(q_y), p=1, dim=-1))
                print(torch.mean(F.normalize(F.relu(q_y), p=1, dim=-1)))
                print("ternary:")
                print(self.ternary)
            self.cnt += 1

        if self.output_prob:
            q_z = self.stepsize * cache_norm_qz + (1-self.stepsize) * self.norm_func(q_z)
        return q_z
    
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
            "output_prob": self.output_prob
        }
        return model_hps
    
    def extra_repr(self) -> str:
        return ", ".join(["{}={}".format(k,v) for k,v in self._get_hyperparams().items()])

class ProbEncoderTD(nn.Module):
    """
    Probabilistic Transformer encoder with tensor decomposition.
    """
    def __init__(self, 
            d_model: int = 256, 
            d_label: int = 64, 
            rank: int = 256, 
            n_iter: int = 4, 
            zero_edge: bool = False, 
            damping: float = 0, 
            stepsize: float = 1, 
            regularize: float = 1,
            norm: str = 'softmax', 
            dists: str = "",
            output_prob: bool = False,
        ):
        """
        Initialize a Probabilistic Transformer encoder with tensor decomposition.
        :param d_model: dimensions of Z nodes.
        :param d_label: dimensions of Y nodes.
        :param rank: rank for tensor decomposition.
        :param n_iter: number of iterations.
        :param zero_edge: whether enforce zero edge between Z nodes.
        :param damping: damping of update. 0 means no damping is applied.
        :param stepsize: step size of update. 1 means full update is applied.
        :param regularize: regularization for MFVI. See 'Regularized Frank-Wolfe for Dense CRFs: 
                           GeneralizingMean Field and Beyond' (Ð.Khuê Lê-Huu, 2021) for details.
        :param norm: normalization method. Options: ['softmax', 'relu'], Default: 'softmax'.
        :param dists: distance pattern. Each distance group will use different factors. 
                      Dists should be groups of numbers seperated by ','. Zero will be excluded.
                      Empty means all tenery factors share the same parameters. Default: "".
                      E.g. "1" -> (-oo, -1], [1, +oo)
                           "-3, -1, 2, 4" -> (-oo, -3), [-3, -1), [-1, 2), [2, 4), [4, +oo)
                                i.e. (-oo, -4], {-3, -2}, {-1, 1}, {2, 3}, [4, +oo)
                           "-2,-1,1,3" -> (-oo, -2), -2, -1, [1, 3), [3, +oo)
        :param output_prob: If true, output a normalized probabilistic distribution. Otherwise
                            output unnormalized scores.
        """
        super().__init__()
        self.d_model = d_model
        self.d_label = d_label
        self.rank = rank
        self.n_iter = n_iter
        self.zero_edge = zero_edge
        self.damping = damping
        self.stepsize = stepsize
        self.regularize = regularize
        self.norm = norm
        self.dists = dists
        self._dists = sorted([int(n) for n in dists.replace(' ', '').split(',') if n])
        self.output_prob = output_prob

        if self.norm == 'softmax':
            self.norm_func = lambda x: F.softmax(x, dim=-1)
        elif self.norm == 'relu':
            self.norm_func = lambda x: F.normalize(F.relu(x), p=1, dim=-1)
        else:
            raise ValueError("%s is not a normalization method." % self.norm)

        self.U = nn.Parameter(torch.Tensor(2, len(self._dists)+1, self.d_model, self.rank))
        self.V = nn.Parameter(torch.Tensor(2, len(self._dists)+1, self.d_model, self.rank))
        self.W = nn.Parameter(torch.Tensor(2, len(self._dists)+1, self.d_label, self.rank))
        ## Seems to have better performance
        nn.init.kaiming_normal_(self.U, nonlinearity='relu')
        nn.init.kaiming_normal_(self.V, nonlinearity='relu')
        nn.init.kaiming_normal_(self.W, nonlinearity='relu')

    def forward(self, x, mask):
        batch_size, max_len, _ = x.shape

        ## Build mask
        mask1d = mask != 0
        mask2d = mask1d.unsqueeze(-1) * mask1d.unsqueeze(-2)
        mask1d = ~mask1d.view(batch_size, max_len, 1)
        mask2d = ~mask2d.view(batch_size, max_len, max_len, 1)

        distmask = torch.ones(len(self._dists)+1, max_len, max_len, dtype=torch.float).to(x.device)
        if len(self._dists) > 0: # At least two dist blocks
            distmask[0] = distmask[0].tril(self._dists[0]-1)
            for i in range(1, len(self._dists)):
                ni_1, ni = self._dists[i-1], self._dists[i]
                distmask[i] = distmask[i].triu(ni_1) - distmask[i].triu(ni)
            distmask[-1] = distmask[-1].triu(self._dists[-1])

        ## Unary score
        unary = x # (batch_size, max_len, d_model)

        ## Zero edge
        if self.zero_edge: 
            with torch.no_grad():
                self.W[:,0,:] = 0

        ## Init with unary score
        q_z = unary.clone()
        q_y = torch.ones(batch_size, max_len, max_len, self.d_label).to(x.device)

        cache_norm_qz, cache_norm_qy = self.norm_func(q_z), self.norm_func(q_y)

        for _ in range(self.n_iter):
            cache_qz, cache_qy = q_z.clone(), q_y.clone()
            
            # Normalize
            q_z, q_y = self.stepsize * cache_norm_qz + (1-self.stepsize) * self.norm_func(q_z), \
                       self.stepsize * cache_norm_qy + (1-self.stepsize) * self.norm_func(q_y)
            
            cache_norm_qz, cache_norm_qy = q_z.clone(), q_y.clone()

            # Apply mask
            q_z = q_z*(~mask1d) 
            q_y = q_y - torch.diag_embed(q_y.diagonal(dim1=1, dim2=2), dim1=1, dim2=2)
            q_y.masked_fill_(mask2d, 0)
            
            second_order_message_F = oe.contract('zjb,zijc,kad,kbd,kcd,kij->zia', *[q_z, q_y, self.U[0], self.V[0], self.W[0], distmask], backend='torch') + \
                                     oe.contract('zjb,zjic,kad,kbd,kcd,kij->zia', *[q_z, q_y, self.U[1], self.V[1], self.W[1], distmask], backend='torch')

            second_order_message_G = oe.contract('zia,zjb,kad,kbd,kcd,kij->zijc',*[q_z, q_z, self.U[0], self.V[0], self.W[0], distmask], backend='torch') + \
                                     oe.contract('zja,zib,kad,kbd,kcd,kij->zijc',*[q_z, q_z, self.U[1], self.V[1], self.W[1], distmask], backend='torch')
            
            # Update
            q_y = cache_qy * self.damping + second_order_message_G * (1-self.damping) / self.regularize
            q_z = cache_qz * self.damping + (unary + second_order_message_F) * (1-self.damping) / self.regularize
                
        if DEBUG:
            if 'cnt' not in self.__dict__:
                self.cnt = 0
            if self.cnt % (2298*16-1) == 0:
                print("Q_z:")
                print(q_z)
                print(torch.mean(q_z))
                print("Q_y:")
                print(q_y)
                print(torch.mean(q_y))
                print("Q_y relu:")
                print(F.normalize(F.relu(q_y), p=1, dim=-1))
                print(torch.mean(F.normalize(F.relu(q_y), p=1, dim=-1)))
            self.cnt += 1

        if self.output_prob:
            q_z = self.stepsize * cache_norm_qz + (1-self.stepsize) * self.norm_func(q_z)
        return q_z

    def _get_hyperparams(self):
        model_hps = {
            "d_model": self.d_model,
            "d_label": self.d_label,
            "rank": self.rank,
            "n_iter": self.n_iter,
            "zero_edge": self.zero_edge,
            "damping": self.damping,
            "stepsize": self.stepsize,
            "regularize": self.regularize,
            "norm": self.norm,
            "dists": self.dists,
            "output_prob": self.output_prob
        }
        return model_hps
    
    def extra_repr(self) -> str:
        return ", ".join(["{}={}".format(k,v) for k,v in self._get_hyperparams().items()])

## #=====================
## FIXME: Deprecated classes

class ProbEncoderWithDistance(ProbEncoder):
    @deprecated(version='1.0.dev', reason="Use ProbEncoder instead.")
    def __init__(self, d_model: int = 32, d_label: int = 32, n_iter: int = 4, zero_edge: bool = False, damping: float = 0, stepsize: float = 1, regularize: float = 1, norm: str = 'softmax', dists: str = "", output_prob: bool = False):
        super().__init__(d_model=d_model, d_label=d_label, n_iter=n_iter, zero_edge=zero_edge, damping=damping, stepsize=stepsize, regularize=regularize, norm=norm, dists=dists, output_prob=output_prob)

class ProbEncoderTDWithDistance(ProbEncoderTD):
    @deprecated(version='1.0.dev', reason="Use ProbEncoderTD instead.")
    def __init__(self, d_model: int = 256, d_label: int = 64, rank: int = 256, n_iter: int = 4, zero_edge: bool = False, damping: float = 0, stepsize: float = 1, regularize: float = 1, norm: str = 'softmax', dists: str = "", output_prob: bool = False):
        super().__init__(d_model=d_model, d_label=d_label, rank=rank, n_iter=n_iter, zero_edge=zero_edge, damping=damping, stepsize=stepsize, regularize=regularize, norm=norm, dists=dists, output_prob=output_prob)

## #=====================
## TODO: Class in development, might merge in the future


## #=====================
## FIXME: Possibly delete the following experimenting class

class ProbEncoderTDFF(nn.Module):
    """
    Probabilistic Transformer encoder with tensor decomposition + global Z label simulating FF layer.
    """
    def __init__(self, d_model=64, d_label=64, d_ff=80, rank=256, n_iter=4, zero_edge=False, damping=0, norm='softmax'):
        """
        Initialize a Probabilistic Transformer encoder with tensor decomposition.
        :param d_model: dimensions of Z nodes.
        :param d_label: dimensions of Y nodes.
        :param d_ff: number of global Z labels.
        :param rank: rank for tensor decomposition.
        :param n_iter: number of iterations.
        :param zero_edge: whether enforce zero edge between Z nodes.
        :param damping: damping of update. 0 means no damping is applied.
        :param norm: normalization method. Options: ['softmax', 'relu'], Default: 'softmax'.
        """
        super().__init__()
        self.d_model = d_model
        self.d_label = d_label
        self.d_ff = d_ff
        self.rank = rank
        self.n_iter = n_iter
        self.zero_edge = zero_edge
        self.damping = damping
        self.norm = norm

        assert norm in ['softmax', 'relu']

        self.U = nn.Parameter(torch.Tensor(2, self.d_model, self.rank))
        self.V = nn.Parameter(torch.Tensor(2, self.d_model, self.rank))
        self.W = nn.Parameter(torch.Tensor(2, self.d_label, self.rank))
        ## Seems to have better performance
        nn.init.kaiming_normal_(self.U, nonlinearity='relu')
        nn.init.kaiming_normal_(self.V, nonlinearity='relu')
        nn.init.kaiming_normal_(self.W, nonlinearity='relu')

        self.ff = nn.Parameter(torch.Tensor(self.d_ff, self.d_model))
        nn.init.kaiming_normal_(self.ff, nonlinearity='relu')

    def forward(self, x, mask):
        batch_size, sent_length, _ = x.shape
        x = torch.cat([x, self.ff.unsqueeze(0).repeat_interleave(batch_size, 0)], dim=1)
        _, max_len, _ = x.shape

        ## Build mask
        mask1d = mask != 0
        mask2d = mask1d.unsqueeze(-1) * mask1d.unsqueeze(-2)
        mask1d = ~mask1d.view(batch_size, max_len, 1)
        mask2d = ~mask2d.view(batch_size, max_len, max_len, 1)

        ## Unary score
        unary = x # (batch_size, max_len, d_model)

        ## Zero edge
        if self.zero_edge: 
            with torch.no_grad():
                self.W[:,0,:] = 0

        ## Init with unary score
        q_z = unary.clone()
        q_y = torch.ones(batch_size, max_len, max_len, self.d_label).to(x.device)

        for _ in range(self.n_iter):
            if self.damping:
              cache_q_z = q_z.clone()
              cache_q_y = q_y.clone()
            
            if self.norm == 'softmax':
                q_z, q_y = q_z.softmax(-1), q_y.softmax(-1)
            elif self.norm == 'relu':
                q_z, q_y = F.normalize(F.relu(q_z), p=1, dim=-1), F.normalize(F.relu(q_y), p=1, dim=-1)

            # Apply mask
            q_z = q_z*(~mask1d) 
            q_y = q_y - torch.diag_embed(q_y.diagonal(dim1=1, dim2=2), dim1=1, dim2=2)
            q_y.masked_fill_(mask2d, 0)
            
            second_order_message = oe.contract('zjb,zijc,ad,bd,cd->zia', *[q_z, q_y, self.U[0], self.V[0], self.W[0]], backend='torch') + \
                                   oe.contract('zjb,zjic,ad,bd,cd->zia', *[q_z, q_y, self.U[1], self.V[1], self.W[1]], backend='torch')
            
            if self.damping:
                q_y = cache_q_y * self.damping + \
                      (oe.contract('zia,zjb,ad,bd,cd->zijc',*[q_z, q_z, self.U[0], self.V[0], self.W[0]], backend='torch') + \
                      oe.contract('zja,zib,ad,bd,cd->zijc',*[q_z, q_z, self.U[1], self.V[1], self.W[1]], backend='torch')) * (1-self.damping)
                q_z = cache_q_z * self.damping + (unary + second_order_message) * (1-self.damping)
            else:
                q_y = oe.contract('zia,zjb,ad,bd,cd->zijc',*[q_z, q_z, self.U[0], self.V[0], self.W[0]], backend='torch') + \
                      oe.contract('zja,zib,ad,bd,cd->zijc',*[q_z, q_z, self.U[1], self.V[1], self.W[1]], backend='torch')
                q_z = unary + second_order_message
                
        if DEBUG:
            if 'cnt' not in self.__dict__:
                self.cnt = 0
            if self.cnt % (2298*16-1) == 0:
                print("Q_z:")
                print(q_z)
                print(torch.mean(q_z))
                print("Q_y:")
                print(q_y)
                print(torch.mean(q_y))
                print("Q_y relu:")
                print(F.normalize(F.relu(q_y), p=1, dim=-1))
                print(torch.mean(F.normalize(F.relu(q_y), p=1, dim=-1)))
            self.cnt += 1

        return q_z[:,:sent_length,:]

    def _get_hyperparams(self):
        model_hps = {
            "d_model": self.d_model,
            "d_label": self.d_label,
            "d_ff": self.d_ff,
            "rank": self.rank,
            "n_iter": self.n_iter,
            "zero_edge": self.zero_edge,
            "damping": self.damping,
            "norm": self.norm
        }
        return model_hps
    
    def extra_repr(self) -> str:
        return ", ".join(["{}={}".format(k,v) for k,v in self._get_hyperparams().items()])