from typing import Dict, List

import torch
import torch.nn as nn

import flair
from flair.data import Dictionary, Sentence, Token
from flair.embeddings import TokenEmbeddings
from flair.models import TextClassifier

from . import modules

class CustomTextClassifier(TextClassifier):
    def __init__(
            self,
            embeddings: TokenEmbeddings,
            tag_dictionary: Dictionary,
            module: dict,
            tag_type: str = None,
            multi_label: bool = None,
            multi_label_threshold: float = 0.5,
            beta: float = 1.0,
            loss_weights: Dict[str, float] = None,
            add_cls: bool = False
    ):
        """
        Initializes a TextClassifier
        :param embeddings: embeddings used to embed each data point
        :param tag_dictionary: dictionary of labels you want to predict
        :param multi_label: auto-detected by default, but you can set this to True to force multi-label prediction
        or False to force single-label prediction
        :param multi_label_threshold: If multi-label you can set the threshold to make predictions
        :param beta: Parameter for F-beta score for evaluation and training annealing
        :param loss_weights: Dictionary of weights for labels for the loss function
        (if any label's weight is unspecified it will default to 1.0)
        :param add_cls: Add [CLS] token to the front of the sentence.
        """

        super(TextClassifier, self).__init__()

        self.embeddings: TokenEmbeddings = embeddings
        self.label_dictionary = self.tag_dictionary = tag_dictionary
        self.label_type = self.tag_type = tag_type

        self.module: dict = module.copy()
        module_class = getattr(modules, module.pop("name"))
        self.mod = module_class(**module)
        self.add_cls = add_cls

        if self.add_cls:
            self.cls_embedding = nn.Parameter(torch.zeros(1, self.embeddings.embedding_length))
        else:
            self.cls_embedding = None

        if multi_label is not None:
            self.multi_label = multi_label
        else:
            self.multi_label = self.tag_dictionary.multi_label

        self.multi_label_threshold = multi_label_threshold

        self.beta = beta

        self.weight_dict = loss_weights
        # Initialize the weight tensor
        if loss_weights is not None:
            n_classes = len(self.tag_dictionary)
            weight_list = [1. for i in range(n_classes)]
            for i, tag in enumerate(self.tag_dictionary.get_items()):
                if tag in loss_weights.keys():
                    weight_list[i] = loss_weights[tag]
            self.loss_weights = torch.FloatTensor(weight_list).to(flair.device)
        else:
            self.loss_weights = None

        self.decoder = nn.Linear(
            module.get("d_root", self.embeddings.embedding_length), len(self.tag_dictionary)
        )

        nn.init.xavier_uniform_(self.decoder.weight)

        if self.multi_label:
            self.loss_function = nn.BCEWithLogitsLoss(weight=self.loss_weights)
        else:
            self.loss_function = nn.CrossEntropyLoss(weight=self.loss_weights)

        # auto-spawn on GPU if available
        self.to(flair.device)

    def forward(self, sentences: List[Sentence]):
        
        self.embeddings.embed(sentences)

        names = self.embeddings.get_names()

        lengths: List[int] = [len(sentence.tokens) for sentence in sentences]
        longest_token_sequence_in_batch: int = max(lengths)

        pre_allocated_zero_tensor = torch.zeros(
            self.embeddings.embedding_length * longest_token_sequence_in_batch,
            dtype=torch.float,
            device=flair.device,
        )

        all_embs = list()
        for sentence in sentences:
            all_embs += [
                emb for token in sentence for emb in token.get_each_embedding(names)
            ]
            nb_padding_tokens = longest_token_sequence_in_batch - len(sentence)

            if nb_padding_tokens > 0:
                t = pre_allocated_zero_tensor[
                    : self.embeddings.embedding_length * nb_padding_tokens
                    ]
                all_embs.append(t)

        sentence_tensor = torch.cat(all_embs).view(
            [
                len(sentences),
                longest_token_sequence_in_batch,
                self.embeddings.embedding_length,
            ]
        )

        weights = torch.arange(
            longest_token_sequence_in_batch,
            device=flair.device
        ).unsqueeze(0) < torch.tensor(
            lengths,
            device=flair.device
        ).unsqueeze(1)
        mask = weights.to(dtype=torch.int32)

        if self.add_cls:
            sentence_tensor = torch.cat((sentence_tensor, self.cls_embedding.repeat_interleave(len(sentences), dim=0).unsqueeze(1)), dim=1)
            mask = torch.cat((mask, torch.ones(len(sentences), 1, device=flair.device)), dim=1)
        
        sentence_tensor = self.mod(sentence_tensor, mask)

        text_embedding_tensor = sentence_tensor[:,0,:]

        label_scores = self.decoder(text_embedding_tensor)

        return label_scores

    def _get_state_dict(self):
        model_state = {
            "state_dict": self.state_dict(),
            "embeddings": self.embeddings,
            "tag_dictionary": self.tag_dictionary,
            "tag_type": self.tag_type,
            "multi_label": self.multi_label,
            "beta": self.beta,
            "weight_dict": self.weight_dict,
            "module": self.module,
            "add_cls": self.add_cls
        }
        return model_state

    @staticmethod
    def _init_model_with_state_dict(state):
        beta = 1.0 if "beta" not in state.keys() else state["beta"]
        weights = None if "weight_dict" not in state.keys() else state["weight_dict"]
        tag_type = None if "tag_type" not in state.keys() else state["tag_type"]

        model = CustomTextClassifier(
            embeddings=state["embeddings"],
            tag_dictionary=state["tag_dictionary"],
            tag_type=tag_type,
            multi_label=state["multi_label"],
            beta=beta,
            loss_weights=weights,
            module=state["module"],
            add_cls=state["add_cls"]
        )

        model.load_state_dict(state["state_dict"])
        return model

    def __str__(self):
        from functools import reduce
        from operator import mul
        total = sum(reduce(mul, v.shape) for k,v in self.state_dict().items())
        module = sum(reduce(mul, v.shape) for k,v in self.mod.state_dict().items())
        return super(flair.nn.Model, self).__str__().rstrip(')') + \
               f'  (beta): {self.beta}\n' + \
               f'  (weights): {self.weight_dict}\n' + \
               f'  (weight_tensor) {self.loss_weights}\n)\n' + \
               f'# Module Parameters: {module}\n' + \
               f'# Total Parameters: {total}\n'