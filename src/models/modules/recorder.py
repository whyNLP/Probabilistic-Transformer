import torch
from supar.structs.fn import chuliu_edmonds

def nonprojective_parse(heads: torch.Tensor):
    """
    Parse with Chu-Liu-Edmonds algorithm.
    Use the Root Reweighting Algorithm to ensure that only one edge coming out from ROOT.
    ref: https://aclanthology.org/2021.emnlp-main.823.pdf

    :param heads: Tensor with shape [length, 1 + length].
    """
    length, _ = heads.shape
    max_, min_ = heads.max(), heads.min()
    c = 1 + length*(max_ - min_)
    
    cloned_heads = heads.clone()
    cloned_heads[:,0] -= c
    scores = torch.cat((torch.zeros(1, 1 + length), cloned_heads), dim=0)
    parse = chuliu_edmonds(scores)

    return parse[1:].numpy().tolist()

class HeadRecorder:
    def __init__(self, use_root = True):
        """
        A recorder to remember the probabilities inside the CRF encoder.
        """
        self.records = {}
        self.use_root = use_root

    def register_batch(self, iteration, probs):
        """
        Register the probabilities for a whole batch.

        :param info: A tuple (iteration, batch, head), index from 0
        :param probs: A normalized tensor with shape (batch_size, num_heads, max_len, 1 + max_len)
                       if not use root, shape (batch_size, num_heads, max_len, max_len)
        """
        batch_size, num_heads, _, _ = probs.shape

        for batch in range(batch_size):
            for head in range(num_heads):
                self.register((iteration, batch, head), probs[batch, head])
    
    def register(self, info, probs):
        """
        Register the probabilities.

        :param info: A tuple (iteration, batch, head), index from 0
        :param probs: A normalized tensor with shape (max_len, 1 + max_len)
                       if not use root, shape (max_len, max_len)
        """
        self.records[info] = probs

    def retrieve_probs(self, iteration, batch = None, head = None):
        """
        Retrieve the probabilities from the records.
        """
        # Resolve negative index
        if iteration < 0:
            iterations = self.iterations
            iteration = iterations[iteration]

        # If only batch size is 1, use this batch.
        if batch is None:
            batches = self.batches
            if len(batches) == 1:
                batch = batches[0]
            else:
                raise ValueError("Batch size is larger than 1. Please choose the batch to retrieve.")
        
        # If head is omitted, return average of all heads.
        if head is None:
            heads = self.heads
            assert len(heads), f"No data for iteration {iteration}, batch {batch}"
            return sum(self.retrieve_probs(iteration, batch, h) for h in heads) / len(heads)

        return self.records[(iteration, batch, head)]

    def __setitem__(self, index, obj):
        """
        Register the probabilities. Shortcut for `register` and `register_batch`.
        """
        if isinstance(index, int):
            self.register_batch(index, obj)
        else:
            self.register(index, obj)

    def __getitem__(self, index):
        """
        Retrieve the probabilities from the records. Short cut for `retrieve_probs`.
        """
        if isinstance(index, int):
            return self.retrieve_probs(index)
        else:
            return self.retrieve_probs(*index)

    @property
    def iterations(self):
        return sorted({iteration for iteration, batch, head in self.records.keys()})

    @property
    def batches(self):
        return sorted({batch for iteration, batch, head in self.records.keys()})

    @property
    def heads(self):
        return sorted({head for iteration, batch, head in self.records.keys()})

    def get_dependency_heads(self, heads, algorithm = 'argmax'):
        """
        Return the dependency heads according to the algorithm.

        :param heads: Tensor with shape [length, length]. Should be normalized.
            if use root, the shape should be [length, 1 + length]
        :param algorithm: Options: 'argmax' (default), 'projective', 'nonprojective'
        :return: confidence: List of confidence of each word's head.
        :return: indices: List of index of each word's head. If use root, the index of root is 0.
        """
        heads = heads.detach().cpu()

        if algorithm == 'argmax':
            confidence, indices = heads.max(dim = 1)
            indices = [i.item() for i in indices] if self.use_root else [i.item()+1 for i in indices]
            confidence = [i.item() for i in confidence]
            return confidence, indices

        assert self.use_root == True, f"Algorithm {algorithm} must use root."

        if algorithm == 'nonprojective':
            indices = nonprojective_parse(heads)
            confidence = [heads[i, h].item() for i, h in enumerate(indices)]
            return confidence, indices
        elif algorithm == 'projective':
            # TODO: Eisner algorithm codes. Or find a module related.
            pass

        raise NotImplementedError(f"Algorithm {algorithm} is not implemented.")

    def export_latex_codes(self, sentence = None, algorithm = 'argmax', mode = 'all'):
        """
        Return the latex codes.
        If the sentence is provided, add to the images.

        :param sentence: The sentence you want to attach to the parse tree.
        :param algorithm: Options: 'argmax' (default), 'projective', 'nonprojective'
        :param mode: Use all heads or average them. Options: 'all' (default), 'average'
        """
        if len(self.records) == 0:
            return ""
        
        import bz2, base64
        HEAD = bz2.decompress(base64.b64decode("QlpoOTFBWSZTWZvzViwAADRfgEAQSgfwEi9Hhs6/79+6MAE6ttmImSGJpqaejRPUaaADTINAZBoSbSmT1PRqDJoAAAAAGmlGmptEA0yAAaGgMmmQX8k5qLDzVSSw+XlBeJCw9qrmMOuwUWrBClf1GWNjac5SyAMPNtO8UR9vFWNxerUhESrdFdqwhC7jZZNZfeUm5uwwnvKCQNObxamVLGizPTBQQ/8bWvFSKLZkX1rYGDEgVzGEKupWjbG/ONL4s2LDxbIPILcGEpCyn9WFawlprI5SdRMMbzSEoGviAIVmUA1SgsThhP9xNAZOrI4ap4oFx3sDAJcNKSyI7YqYGXBMQ0Pi/FGJJjFC4ogicqoCr6sS+T0Yp4+kboSD3vKzRRLIQbOaj58hRDOimUGpYmNEDo1OTGKaSi0pmCzU5yNa0HM0RGdbV16yiQ2TUA7wkJj0Mygo1J42YQryoSAOQTjYSUNIzcSIZUjtQLaaxfT/F3JFOFCQm/NWLA==")).decode()
        TAIL = '\\end{document}'

        iterations, batches, heads = self.iterations, self.batches, self.heads

        codes = ''
        if len(batches) > 1:
            for batch in batches:
                codes += '\\section{Batch ' + str(batch+1) + '}\n'
                for iteration in iterations:
                    codes += '\\subsection{Iteration ' + str(iteration+1) + '}\n'
                    if mode == 'average':
                        codes += self.draw_latex_head(self[iteration, batch], algorithm)
                    else:
                        for head in heads:
                            codes += '\\subsubsection{Channel ' + str(head+1) + '}\n'
                            codes += self.draw_latex_head(self[iteration, batch, head], algorithm)
        else:
            for iteration in iterations:
                codes += '\\section{Iteration ' + str(iteration+1) + '}\n'
                if mode == 'average':
                    codes += self.draw_latex_head(self[iteration], algorithm)
                else:
                    for head in heads:
                        codes += '\\subsection{Channel ' + str(head+1) + '}\n'
                        codes += self.draw_latex_head(self.retrieve_probs(iteration, head=head), algorithm)
        
            if sentence:
                if isinstance(sentence, str):
                    sentence = sentence.split()
                elif isinstance(sentence, list):
                    pass
                else: # flair Sentence object
                    sentence = [token.text for token in sentence]
                codes = codes.replace("begin{deptext}[column sep=1em]", "begin{deptext}[column sep=1em]\n        " + ' \\& '.join(sentence) + ' \\\\')

        return HEAD + codes + TAIL

    def draw_latex_head(self, heads, algorithm = 'argmax'):
        """
        Generate the latex codes to draw the dependency parse tree, given the
        input head matrix.

        :param heads: Tensor with shape [length, length]. Should be normalized.
        :rtype: str: Latex codes to draw the dependency parse tree.
        """
        ## Move the tensor to cpu
        confidence, indices = self.get_dependency_heads(heads, algorithm)
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
            idx = indices[i]
            conf = confidence[i]
            if self.use_root:
                if idx == 0:
                    s += '    \\deproot{{{}}}{{ROOT ({:.2f})}}\n'.format(str(i+1), conf)
                else:
                    s += '    \\depedge{{{}}}{{{}}}{{{:.2f}}}\n'.format(str(idx), str(i+1), conf)
            else:
                s += '    \\depedge{{{}}}{{{}}}{{{:.2f}}}\n'.format(str(idx), str(i+1), conf)
        
        s += '\\end{dependency}\n\n'

        return s


if __name__ == '__main__':
    recoder = HeadRecorder()
    recoder[0,0,0] = torch.tensor([
        [.2, 0., .5, .3],
        [.1, .2, 0., .7],
        [.8, .1, .1, 0.],
    ])
    recoder[0,0,1] = torch.tensor([
        [.2, 0., .3, .5],
        [.2, .1, 0., .7],
        [.2, .4, .4, 0.],
    ])
    # print(recoder[0,0,0])
    s = recoder.export_latex_codes("v1 v2 v3", 'nonprojective', 'average')
    print(s)

    # a = torch.tensor([
    #     [5, 0, 10, 8],
    #     [1, 11, 0, 8],
    #     [1, 4, 5, 0]
    # ])
    # r = nonprojective_parse(a)
    # print(r)