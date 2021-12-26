from collections import Counter

import torch
import flair
import logging

from flair.data import Sentence, Token, Corpus, Dictionary
from flair.embeddings.base import Embeddings, ScalarMix
from flair.embeddings import OneHotEmbeddings

log = logging.getLogger("flair")

class MLMOneHotEmbeddings(OneHotEmbeddings):
    """One-hot encoded embeddings with <MASK>."""

    def __init__(
            self,
            corpus: Corpus,
            field: str = "text",
            embedding_length: int = 300,
            min_freq: int = 3,
    ):
        """
        Initializes one-hot encoded word embeddings and a trainable embedding layer
        :param corpus: you need to pass a Corpus in order to construct the vocabulary
        :param field: by default, the 'text' of tokens is embedded, but you can also embed tags such as 'pos'
        :param embedding_length: dimensionality of the trainable embedding layer
        :param min_freq: minimum frequency of a word to become part of the vocabulary
        """
        super(OneHotEmbeddings, self).__init__()
        self.name = "one-hot-mlm"
        self.static_embeddings = False
        self.min_freq = min_freq
        self.field = field
        self.instance_parameters = self.get_instance_parameters(locals=locals())

        tokens = list(map((lambda s: s.tokens), corpus.train))
        tokens = [token for sublist in tokens for token in sublist]

        if field == "text":
            most_common = Counter(list(map((lambda t: t.text), tokens))).most_common()
        else:
            most_common = Counter(
                list(map((lambda t: t.get_tag(field).value), tokens))
            ).most_common()

        tokens = []
        for token, freq in most_common:
            if freq < min_freq:
                break
            tokens.append(token)

        self.vocab_dictionary: Dictionary = Dictionary()
        for token in tokens:
            self.vocab_dictionary.add_item(token)
        self.vocab_dictionary.add_item("<MASK>")

        # max_tokens = 500
        self.__embedding_length = embedding_length

        print(f"vocabulary size of {len(self.vocab_dictionary)}")

        # model architecture
        self.embedding_layer = torch.nn.Embedding(
            len(self.vocab_dictionary), self.__embedding_length
        )
        torch.nn.init.xavier_uniform_(self.embedding_layer.weight)

        self.to(flair.device)

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length