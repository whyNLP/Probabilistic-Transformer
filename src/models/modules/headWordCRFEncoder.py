import torch
import torch.nn as nn
import torch.nn.functional as F

import opt_einsum as oe
import math

import os
DEBUG=os.environ.get('DEBUG')
DRAW=os.environ.get('DRAW')

class HeadWordProbEncoder(nn.Module):
    """
    Head-Selection Word Probabilistic Transformer encoder.
    """
    def __init__(self, 
            d_model: int = 32, 
            n_head: int = 10, 
            n_iter: int = 4, 
            damping_H: float = 0, 
            damping_Z: float = 0, 
            damping_X: float = 0, 
            stepsize_H: float = 1, 
            stepsize_Z: float = 1, 
            stepsize_X: float = 1, 
            regularize_H: float = 1,
            regularize_Z: float = 1,
            regularize_X: float = 1,
            norm: str = 'softmax', 
            dists: str = "",
            async_update: bool = True,
            output_prob: bool = False,
            use_td: str = 'no',
            dropout: float = 0,
            block_msg: bool = False,
            initialize_Z: bool = True
        ):
        """
        Initialize a basic Probabilistic Transformer encoder.
        :param d_model: dimensions of Z nodes.
        :param n_head: number of heads.
        :param n_iter: number of iterations.
        :param damping_H: damping of H nodes update. 0 means no damping is applied.
        :param damping_Z: damping of Z nodes update. 0 means no damping is applied.
        :param damping_X: damping of X nodes update. 0 means no damping is applied.
        :param stepsize_H: step size of H nodes update. 1 means full update is applied.
        :param stepsize_Z: step size of Z nodes update. 1 means full update is applied.
        :param stepsize_X: step size of X nodes update. 1 means full update is applied.
        :param regularize_H: regularization for updating H nodes.
        :param regularize_Z: regularization for updating Z nodes.
        :param regularize_X: regularization for updating X.
                'regularize_H', 'regularize_X' and 'regularize_Z' are regularizations
                for MFVI. See 'Regularized Frank-Wolfe for Dense CRFs: Generalizing
                Mean Field and Beyond' (Ð.Khuê Lê-Huu, 2021) for details.
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
        :param initialize_Z: initialize masked Z nodes with X nodes. Default: True. 
        """
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.n_iter = n_iter
        self.damping_H = damping_H
        self.damping_Z = damping_Z
        self.damping_X = damping_X
        self.stepsize_H = stepsize_H
        self.stepsize_Z = stepsize_Z
        self.stepsize_X = stepsize_X
        self.regularize_H = regularize_H
        self.regularize_Z = regularize_Z
        self.regularize_X = regularize_X
        self.norm = norm
        self.dists = dists
        self._dists = sorted([int(n) for n in dists.replace(' ', '').split(',') if n])
        self.async_update = async_update
        self.output_prob = output_prob
        self.use_td = use_td.replace(' ', '')
        self.dropout = dropout
        self.block_msg = block_msg
        self.initialize_Z = initialize_Z

        assert not self._dists or all([n>1 for n in self._dists]), "The minimum seperate point should be 2. See docs about distance."

        if self.norm == 'softmax':
            self.norm_func = lambda x: F.softmax(x, dim=-1)
        elif self.norm == 'relu':
            self.norm_func = lambda x: F.normalize(F.relu(x), p=1, dim=-1)
        else:
            raise ValueError("%s is not a normalization method." % self.norm)

        if self.use_td.startswith('uv:'):
            rank = int(self.use_td.split(':')[-1])
            self.U = nn.Parameter(torch.Tensor(2*(len(self._dists)+1), self.d_model, rank, self.n_head))
            self.V = nn.Parameter(torch.Tensor(2*(len(self._dists)+1), self.d_model, rank, self.n_head))
            nn.init.normal_(self.U, std = 1/math.sqrt(rank))
            nn.init.normal_(self.V, std = 1/math.sqrt(rank))
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
            ternary = oe.contract('kadc,kbdc->kabc', *[self.U, self.V], backend='torch')
        elif self.use_td.startswith('uvw:'):
            ternary = oe.contract('kad,kbd,kcd->kabc', *[self.U, self.V, self.W], backend='torch')
        else:
            ternary = self.ternary

        ## Init with unary score
        q_h = torch.ones(batch_size, self.n_head, max_len, max_len).to(x.device)
        q_x = torch.ones(batch_size, max_len, vocab_size, dtype=torch.float).to(x.device)

        if self.initialize_Z:
            pre_allocated_ones = oe.contract('zix,xa->zia', *[self.norm_func(q_x), embed_matrix], backend='torch')
        else:
            pre_allocated_ones = torch.ones(batch_size, max_len, self.d_model, dtype=torch.float).to(x.device)
        
        q_z = unary.clone() * (1-m_mask).unsqueeze(-1) + pre_allocated_ones * m_mask.unsqueeze(-1)
               
        # Apply mask
        q_z = q_z*(~mask1d)
        q_h = q_h - torch.diag_embed(torch.ones_like(q_h[...,0])*(1e9), dim1=-1, dim2=-2)
        q_h.masked_fill_(mask2d, -1e9)

        ## Initialization for async Y nodes
        cache_qh = q_h.clone()

        cache_norm_qz, cache_norm_qh, cache_norm_qx = self.norm_func(q_z), self.norm_func(q_h), self.norm_func(q_x)

        ## In async mode, only q_x is unnormalized.
        if self.async_update:
            cache_qh, cache_qz= q_h.clone(), q_z.clone()
            q_z = self.norm_func(q_z)
            q_z = q_z*(~mask1d)

        for iteration in range(self.n_iter):

            if self.async_update:
                
                ## Update H first
                # Calculate 2nd message for different dists
                second_order_message_F = oe.contract('zia,zjb,kabc,kij->zcij',*[q_z, q_z, ternary, distmask], backend='torch')
                
                # Update
                q_h = cache_qh * self.damping_H + second_order_message_F * (1-self.damping_H) / self.regularize_H * self.d_model

                # Apply mask
                q_h = q_h - torch.diag_embed(torch.ones_like(q_h[...,0])*(1e9), dim1=-1, dim2=-2)
                q_h.masked_fill_(mask2d, -1e9)
            
                ## Then update Z
                cache_qh, cache_qx = q_h.clone(), q_x.clone()
                
                # Normalize
                q_h = ( (1-self.stepsize_H) * cache_norm_qh + self.stepsize_H * self.norm_func(q_h) ) if iteration else self.norm_func(q_h)
                q_x = ( (1-self.stepsize_X) * cache_norm_qx + self.stepsize_X * self.norm_func(q_x) ) if iteration else self.norm_func(q_x)
                q_h = self.dropout_h(q_h)
                cache_norm_qh, cache_norm_qx = q_h.clone(), q_x.clone()
                
                # Calculate 2nd message for different dists
                second_order_message_G = oe.contract('zjb,zcij,kabc,kij->zia', *[q_z, q_h, ternary, distmask], backend='torch')
                if not self.block_msg:
                    second_order_message_G = second_order_message_G + oe.contract('zjb,zcji,kbac,kji->zia', *[q_z, q_h, ternary, distmask], backend='torch')
                second_order_message_H = oe.contract('zix,xa->zia', *[q_x, embed_matrix], backend='torch')

                # Update
                q_z = cache_qz * self.damping_Z + (
                    (unary + second_order_message_G) * (1-m_mask).unsqueeze(-1) + (second_order_message_H + second_order_message_G) * m_mask.unsqueeze(-1)
                ) * (1-self.damping_Z) / self.regularize_Z
                q_z = self.dropout_z(q_z)

                ## Finally update X
                cache_qz = q_z.clone()
                
                # Normalize
                q_z = ( (1-self.stepsize_Z) * cache_norm_qz + self.stepsize_Z * self.norm_func(q_z) ) if iteration else self.norm_func(q_z)
                cache_norm_qz = q_z.clone()
                
                # Apply mask
                q_z = q_z*(~mask1d) 

                # Calculate 2nd message
                second_order_message_I = oe.contract('zia,xa->zix', *[q_z, embed_matrix], backend='torch')

                # Update
                q_x = cache_qx * self.damping_X + second_order_message_I * (1-self.damping_X) / self.regularize_X
            
            else:

                raise NotImplementedError

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
                if not iteration: codes = '% In the preamble:\n\\usepackage{tikz-dependency}\n\n% In the document:\n'
                codes += '\\section{Iteration ' + str(iteration+1) + '}\n'
                for b in range(batch_size):
                    codes += '\\subsection{Batch ' + str(b+1) + '}\n'
                    for h in range(self.n_head):
                        codes += '\\subsubsection{Head ' + str(h+1) + '}\n'
                        codes += self.draw_latex_dep(q_h[b,h])
                if iteration + 1 == self.n_iter:
                    print(codes)
                    exit(0)
                
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
                print(ternary)
                print(torch.mean(ternary))
                print(torch.std(ternary))
                exit(0)
            
            self.cnt += 1
        
        # Save the Q value for H nodes
        if not self.async_update:
            q_h = (1-self.stepsize_H) * cache_norm_qh + self.stepsize_H * self.norm_func(q_h)
            q_h = q_h - torch.diag_embed(q_h.diagonal(dim1=1, dim2=2), dim1=1, dim2=2)
            q_h.masked_fill_(mask2d, 0)
        self.q_h = q_h

        if self.output_prob:
            q_x = (1-self.stepsize_X) * cache_norm_qx + self.stepsize_X * self.norm_func(q_x)
        return q_x
    
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
            "damping_X": self.damping_X,
            "stepsize_H": self.stepsize_H,
            "stepsize_Z": self.stepsize_Z,
            "stepsize_X": self.stepsize_X,
            "regularize_H": self.regularize_H,
            "regularize_Z": self.regularize_Z,
            "regularize_X": self.regularize_X,
            "norm": self.norm,
            "dists": self.dists,
            "async_update": self.async_update,
            "output_prob": self.output_prob,
            'use_td': self.use_td,
            "dropout": self.dropout,
            "block_msg": self.block_msg,
            "initialize_Z": self.initialize_Z
        }
        return model_hps
    
    def extra_repr(self) -> str:
        return ", ".join(["{}={}".format(k,v) for k,v in self._get_hyperparams().items()])

    @classmethod
    def draw_latex_dep(cls, heads):
        """
        Generate the latex codes to draw the dependency parse tree, given the
        input head matrix.

        :param: heads: Tensor with shape [length, length]. Should be normalized.
        :rtype: str: Latex codes to draw the dependency parse tree.
        """
        ## Move the tensor to cpu
        heads = heads.detach().cpu()
        confidence, indices = heads.max(dim = 1)
        length = len(heads)

        ## The head code
        s = """
\\begin{dependency}[theme = simple]
    \\begin{deptext}[column sep=1em]
        """
        s += ' \\& '.join([str(i+1) for i in range(length)])
        s += """ \\\\
    \\end{deptext}
"""

        ## The dependency edges
        for i in range(length):
            idx = indices[i].item()
            conf = confidence[i].item()
            s += '    \\depedge{{{}}}{{{}}}{{{:.2f}}}\n'.format(str(idx+1), str(i+1), conf)
        
        s += '\\end{dependency}\n\n'

        return s


class PseudoHeadWordProbEncoder(HeadWordProbEncoder):
    """
    Pseudo Head-Selection Word Probabilistic Transformer encoder.

    We only update X node after all the iterations -- X nodes are detached during 
    the inference. damping_X and stepsize_X will not work.
    """

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
            ternary = oe.contract('kadc,kbdc->kabc', *[self.U, self.V], backend='torch')
        elif self.use_td.startswith('uvw:'):
            ternary = oe.contract('kad,kbd,kcd->kabc', *[self.U, self.V, self.W], backend='torch')
        else:
            ternary = self.ternary

        ## Init with unary score
        q_h = torch.ones(batch_size, self.n_head, max_len, max_len).to(x.device)
        q_x = torch.ones(batch_size, max_len, vocab_size, dtype=torch.float).to(x.device)

        if self.initialize_Z:
            pre_allocated_ones = oe.contract('zix,xa->zia', *[self.norm_func(q_x), embed_matrix], backend='torch')
        else:
            pre_allocated_ones = torch.ones(batch_size, max_len, self.d_model, dtype=torch.float).to(x.device)
        
        q_z = unary.clone() * (1-m_mask).unsqueeze(-1) + pre_allocated_ones * m_mask.unsqueeze(-1)
               
        # Apply mask
        q_z = q_z*(~mask1d)
        q_h = q_h - torch.diag_embed(torch.ones_like(q_h[...,0])*(1e9), dim1=-1, dim2=-2)
        q_h.masked_fill_(mask2d, -1e9)

        ## Initialization for async Y nodes
        cache_qh = q_h.clone()

        cache_norm_qz, cache_norm_qh, cache_norm_qx = self.norm_func(q_z), self.norm_func(q_h), self.norm_func(q_x)

        ## In async mode, only q_x is unnormalized.
        if self.async_update:
            cache_qh, cache_qz= q_h.clone(), q_z.clone()
            q_z = self.norm_func(q_z)
            q_z = q_z*(~mask1d)

        for iteration in range(self.n_iter):

            if self.async_update:
                
                ## Update H first
                # Calculate 2nd message for different dists
                second_order_message_F = oe.contract('zia,zjb,kabc,kij->zcij',*[q_z, q_z, ternary, distmask], backend='torch')
                
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
                second_order_message_G = oe.contract('zjb,zcij,kabc,kij->zia', *[q_z, q_h, ternary, distmask], backend='torch')
                if not self.block_msg:
                    second_order_message_G = second_order_message_G + oe.contract('zjb,zcji,kbac,kji->zia', *[q_z, q_h, ternary, distmask], backend='torch')

                # Update
                q_z = cache_qz * self.damping_Z + (
                    (unary + second_order_message_G) * (1-m_mask).unsqueeze(-1) + (second_order_message_G) * m_mask.unsqueeze(-1)
                ) * (1-self.damping_Z) / self.regularize_Z
                q_z = self.dropout_z(q_z)

                ## Finally update X
                cache_qz = q_z.clone()
                
                # Normalize
                q_z = ( (1-self.stepsize_Z) * cache_norm_qz + self.stepsize_Z * self.norm_func(q_z) ) if iteration else self.norm_func(q_z)
                cache_norm_qz = q_z.clone()
                
                # Apply mask
                q_z = q_z*(~mask1d) 
            
            else:

                raise NotImplementedError

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
                if not iteration: codes = '% In the preamble:\n\\usepackage{tikz-dependency}\n\n% In the document:\n'
                codes += '\\section{Iteration ' + str(iteration+1) + '}\n'
                for b in range(batch_size):
                    codes += '\\subsection{Batch ' + str(b+1) + '}\n'
                    for h in range(self.n_head):
                        codes += '\\subsubsection{Head ' + str(h+1) + '}\n'
                        codes += self.draw_latex_dep(q_h[b,h])
                if iteration + 1 == self.n_iter:
                    print(codes)
                    exit(0)

        # Calculate 2nd message
        second_order_message_I = oe.contract('zia,xa->zix', *[q_z, embed_matrix], backend='torch')

        # Update
        q_x = second_order_message_I / self.regularize_X
                
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
                print(ternary)
                print(torch.mean(ternary))
                print(torch.std(ternary))
                exit(0)
            
            self.cnt += 1
        
        # Save the Q value for H nodes
        if not self.async_update:
            q_h = (1-self.stepsize_H) * cache_norm_qh + self.stepsize_H * self.norm_func(q_h)
            q_h = q_h - torch.diag_embed(q_h.diagonal(dim1=1, dim2=2), dim1=1, dim2=2)
            q_h.masked_fill_(mask2d, 0)
        self.q_h = q_h

        if self.output_prob:
            q_x = self.norm_func(q_x)
        return q_x
