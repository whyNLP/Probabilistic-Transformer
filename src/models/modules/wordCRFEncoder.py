import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import opt_einsum as oe

import os
DEBUG=os.environ.get('DEBUG')

class WordProbEncoder(nn.Module):
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
            async_update: bool = False,
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
                      Dists should be groups of numbers seperated by ','. Each number represents
                      a seperate point. Empty means all tenery factors share the same parameters.
                      Note that the minimum seperate point you input should be 2. Default: "".
                      E.g. "" -> [1, +oo)
                           "3" -> [1, 2), [3, +oo)
                           "2, 4" -> [1, 2), [2, 4), [4, +oo)
                                i.e. {1}, {2, 3}, [4, +oo)
        :param async_update: update the q values asyncronously (Y first, then Z).
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
        self.async_update = async_update
        self.output_prob = output_prob

        assert not self._dists or all([n>1 for n in self._dists]), "The minimum seperate point should be 2. See docs about distance."

        if self.norm == 'softmax':
            self.norm_func = lambda x: F.softmax(x, dim=-1)
        elif self.norm == 'relu':
            self.norm_func = lambda x: F.normalize(F.relu(x), p=1, dim=-1)
        else:
            raise ValueError("%s is not a normalization method." % self.norm)

        self.ternary = nn.Parameter(torch.ones(2, len(self._dists)+1, self.d_model, self.d_model, self.d_label))
        nn.init.normal_(self.ternary)

    def forward(self, x, mask, embed_matrix, m_mask):
        """
        :param x: input tensor with shape [batch_size, max_length, d_model]
        :param mask: token mask (1 for word and 0 for pad) with shape [batch_size, max_length] dtype=torch.int32
        :param embed_matrix: the embedding matrix [vocab_size, d_model]
        :param m_mask: <MASK> mask (1 for <MASK> and 0 for others) with shape [batch_size, max_length] dtype=torch.int32

        :return q_x: normalized q value of x variables. tensor with shape [batch_size, max_length, vocab_size]
        """
        batch_size, max_len, _ = x.shape
        vocab_size, _ = embed_matrix.shape

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
        
        pre_allocated_ones = torch.ones(batch_size, max_len, self.d_model, dtype=torch.float).to(x.device)

        ## Unary score
        unary = x # (batch_size, max_len, d_model)

        ## Zero edge
        if self.zero_edge:
            with torch.no_grad():
                self.ternary[..., 0] = 0

        ## Init with unary score
        q_z = unary.clone() * (1-m_mask).unsqueeze(-1) + pre_allocated_ones * m_mask.unsqueeze(-1)
        q_y = torch.ones(batch_size, max_len, max_len, self.d_label).to(x.device)
        q_x = torch.ones(batch_size, max_len, vocab_size, dtype=torch.float).to(x.device)

        cache_norm_qz, cache_norm_qy, cache_norm_qx = self.norm_func(q_z), self.norm_func(q_y), self.norm_func(q_x)

        ## In async mode, only q_x is unnormalized.
        if self.async_update:
            cache_qy, cache_qz= q_y.clone(), q_z.clone()
            q_z = self.norm_func(q_z)
            q_z = q_z*(~mask1d)

        for _ in range(self.n_iter):

            if self.async_update:

                ## Update Y first
                # Calculate 2nd message for different dists
                second_order_message_G = oe.contract('zia,zjb,kabc,kij->zijc',*[q_z, q_z, self.d_model * self.ternary[0], distmask], backend='torch') + \
                                         oe.contract('zia,zjb,kbac,kji->zijc',*[q_z, q_z, self.d_model * self.ternary[1], distmask], backend='torch')
                
                # Update
                q_y = cache_qy * self.damping + second_order_message_G * (1-self.damping) / self.regularize
            
                ## Then update Z
                cache_qy, cache_qx = q_y.clone(), q_x.clone()
                
                # Normalize
                q_y = (1-self.stepsize) * cache_norm_qy + self.stepsize * self.norm_func(q_y)
                q_x = (1-self.stepsize) * cache_norm_qx + self.stepsize * self.norm_func(q_x)
                cache_norm_qy, cache_norm_qx = q_y.clone(), q_x.clone()
                
                # Apply mask
                q_y = q_y - torch.diag_embed(q_y.diagonal(dim1=1, dim2=2), dim1=1, dim2=2)
                q_y.masked_fill_(mask2d, 0)
                q_x = q_x * m_mask.unsqueeze(-1)
                
                # Calculate 2nd message for different dists
                second_order_message_F = oe.contract('zjb,zijc,kabc,kij->zia', *[q_z, q_y, self.d_model * self.ternary[0], distmask], backend='torch') + \
                                         oe.contract('zjb,zjic,kabc,kij->zia', *[q_z, q_y, self.d_model * self.ternary[1], distmask], backend='torch') + \
                                         oe.contract('zjb,zjic,kbac,kji->zia', *[q_z, q_y, self.d_model * self.ternary[0], distmask], backend='torch') + \
                                         oe.contract('zjb,zijc,kbac,kji->zia', *[q_z, q_y, self.d_model * self.ternary[1], distmask], backend='torch')
                
                second_order_message_H = oe.contract('zix,xa->zia', *[q_x, embed_matrix], backend='torch')
                
                # Update
                q_z = cache_qz * self.damping + (
                    (unary + second_order_message_F) * (1-m_mask).unsqueeze(-1) + (second_order_message_H + second_order_message_F) * m_mask.unsqueeze(-1)
                ) * (1-self.damping) / self.regularize / self.d_model

                ## Finally update X
                cache_qz = q_z.clone()
                
                # Normalize
                q_z = (1-self.stepsize) * cache_norm_qz + self.stepsize * self.norm_func(q_z)
                cache_norm_qz = q_z.clone()
                
                # Apply mask
                q_z = q_z*(~mask1d) 

                # Calculate 2nd message
                second_order_message_I = oe.contract('zia,xa->zix', *[q_z, embed_matrix], backend='torch')

                # Update
                q_x = cache_qx * self.damping + second_order_message_I * (1-self.damping) / self.regularize * math.sqrt(self.d_model)

            else:

                cache_qz, cache_qy, cache_qx = q_z.clone(), q_y.clone(), q_x.clone()
                
                # Normalize
                q_z, q_y, q_x = (1-self.stepsize) * cache_norm_qz + self.stepsize * self.norm_func(q_z), \
                                (1-self.stepsize) * cache_norm_qy + self.stepsize * self.norm_func(q_y), \
                                (1-self.stepsize) * cache_norm_qx + self.stepsize * self.norm_func(q_x)
                
                cache_norm_qz, cache_norm_qy, cache_norm_qx = q_z.clone(), q_y.clone(), q_x.clone()
                
                # Apply mask
                q_z = q_z*(~mask1d) 
                q_y = q_y - torch.diag_embed(q_y.diagonal(dim1=1, dim2=2), dim1=1, dim2=2)
                q_y.masked_fill_(mask2d, 0)
                q_x = q_x * m_mask.unsqueeze(-1)
                
                # Calculate 2nd message for different dists
                second_order_message_F = oe.contract('zjb,zijc,kabc,kij->zia', *[q_z, q_y, self.ternary[0], distmask], backend='torch') + \
                                         oe.contract('zjb,zjic,kabc,kij->zia', *[q_z, q_y, self.ternary[1], distmask], backend='torch') + \
                                         oe.contract('zjb,zjic,kbac,kji->zia', *[q_z, q_y, self.ternary[0], distmask], backend='torch') + \
                                         oe.contract('zjb,zijc,kbac,kji->zia', *[q_z, q_y, self.ternary[1], distmask], backend='torch')

                second_order_message_G = oe.contract('zia,zjb,kabc,kij->zijc',*[q_z, q_z, self.ternary[0], distmask], backend='torch') + \
                                         oe.contract('zia,zjb,kbac,kji->zijc',*[q_z, q_z, self.ternary[1], distmask], backend='torch')

                second_order_message_H = oe.contract('zix,xa->zia', *[q_x, embed_matrix], backend='torch')

                second_order_message_I = oe.contract('zia,xa->zix', *[q_z, embed_matrix], backend='torch')
                
                # Update
                q_y = cache_qy * self.damping + second_order_message_G * (1-self.damping) / self.regularize
                q_z = cache_qz * self.damping + (
                    (unary + second_order_message_F) * (1-m_mask).unsqueeze(-1) + (second_order_message_H + second_order_message_F) * m_mask.unsqueeze(-1)
                ) * (1-self.damping) / self.regularize
                q_x = cache_qx * self.damping + second_order_message_I * (1-self.damping) / self.regularize
                
        if DEBUG:
            if 'cnt' not in self.__dict__:
                self.cnt = 0
            if self.cnt % (int(DEBUG)) == (int(DEBUG)-1):
                print("Q_z:")
                print(q_z)
                print(torch.mean(q_z))
                print(torch.std(q_z))
                print("Q_z norm:")
                print(self.norm_func(q_z))
                print(torch.mean(self.norm_func(q_z)))
                print("Q_y:")
                print(q_y)
                print(torch.mean(q_y))
                print(torch.std(q_y))
                print("Q_y norm:")
                print(self.norm_func(q_y))
                print(torch.mean(self.norm_func(q_y)))
                print("Q_x:")
                print(q_x)
                print(torch.mean(q_x))
                print(torch.std(q_x))
                print("Q_x norm:")
                print(self.norm_func(q_x))
                print(torch.mean(self.norm_func(q_x)))
                print("embed_matrix:")
                print(embed_matrix)
                print(torch.mean(embed_matrix))
                print(torch.std(embed_matrix))
                print("ternary:")
                print(self.ternary)
                exit(0)
            self.cnt += 1
        
        # Save the Q value for Y nodes
        if not self.async_update:
            q_y = (1-self.stepsize) * cache_norm_qy + self.stepsize * self.norm_func(q_y)
            q_y = q_y - torch.diag_embed(q_y.diagonal(dim1=1, dim2=2), dim1=1, dim2=2)
            q_y.masked_fill_(mask2d, 0)
        self.q_y = q_y

        if self.output_prob:
            q_x = (1-self.stepsize) * cache_norm_qx + self.stepsize * self.norm_func(q_x)
        return q_x
    
    def getYNorm(self, p):
        return (1-self.q_y[..., 0]).norm(p=p)
    
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
            "output_prob": self.output_prob
        }
        return model_hps
    
    def extra_repr(self) -> str:
        return ", ".join(["{}={}".format(k,v) for k,v in self._get_hyperparams().items()])

## #=====================
## TODO: Class in development, might merge in the future

class WordProbEncoderTD(nn.Module):
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
            if self.cnt % (int(DEBUG)) == (int(DEBUG)-1):
                print("Q_z:")
                print(q_z)
                print(torch.mean(q_z))
                print("Q_y:")
                print(q_y)
                print(torch.mean(q_y))
                print("Q_y norm:")
                print(self.norm_func(q_y))
                print(torch.mean(self.norm_func(q_y)))
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