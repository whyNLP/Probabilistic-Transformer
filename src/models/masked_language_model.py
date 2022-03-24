from flair.embeddings.token import OneHotEmbeddings
from .sequence_tagger import CustomSequenceTagger

import logging
import sys
import math

from pathlib import Path
from typing import List, Union, Optional, Dict, Tuple
from warnings import warn

import numpy as np
import torch
import torch.nn
import torch.nn.functional as F
from requests import HTTPError
from tabulate import tabulate
from torch.nn.parameter import Parameter
from torch.utils.data.dataset import Dataset
from tqdm import tqdm

import flair.nn
from flair.data import Dictionary, Sentence, Label
from flair.datasets import SentenceDataset, DataLoader
from flair.models import SequenceTagger
from flair.embeddings import TokenEmbeddings, StackedEmbeddings, Embeddings
from flair.file_utils import cached_path, unzip_file
from flair.training_utils import Metric, Result, store_embeddings

from . import modules

log = logging.getLogger("flair")

class MaskedLanguageModel(CustomSequenceTagger):

    def __init__(
            self,
            embeddings: TokenEmbeddings,
            tag_dictionary: Dictionary,
            tag_type: str,
            module: dict,
            hidden_size: int = 256,
            use_crf: bool = False,
            use_rnn: bool = False,
            rnn_layers: int = 1,
            dropout: float = 0.0,
            word_dropout: float = 0.0,
            locked_dropout: float = 0.0,
            reproject_embeddings: Union[bool, int] = False,
            reuse_embedding_weight: bool = False,
            train_initial_hidden_state: bool = False,
            rnn_type: str = "LSTM",
            pickle_module: str = "pickle",
            beta: float = 1.0,
            loss_weights: Dict[str, float] = None,
    ):
        """
        Initializes a SequenceTagger
        :param hidden_size: number of hidden states in RNN
        :param embeddings: word embeddings used in tagger
        :param tag_dictionary: dictionary of tags you want to predict
        :param tag_type: string identifier for tag type
        :param use_crf: if True use CRF decoder, else project directly to tag space
        :param use_rnn: if True use RNN layer, otherwise use word embeddings directly
        :param rnn_layers: number of RNN layers
        :param dropout: dropout probability
        :param word_dropout: word dropout probability
        :param reproject_embeddings: if True, adds trainable linear map on top of embedding layer. If False, no map.
        If you set this to an integer, you can control the dimensionality of the reprojection layer
        :param locked_dropout: locked dropout probability
        :param train_initial_hidden_state: if True, trains initial hidden state of RNN
        :param beta: Parameter for F-beta score for evaluation and training annealing
        :param loss_weights: Dictionary of weights for classes (tags) for the loss function
        :param module: Embedding module which is gona to use. A dict specifies name and hyperparameters:
        e.g. {
                "name": "...",
                "d_label": "..."
             }
        (if any tag's weight is unspecified it will default to 1.0)

        """

        super(SequenceTagger, self).__init__()
        self.use_rnn = use_rnn
        self.hidden_size = hidden_size
        self.use_crf: bool = use_crf
        self.rnn_layers: int = rnn_layers

        self.trained_epochs: int = 0

        self.embeddings = embeddings
        self.reuse_embedding_weight = reuse_embedding_weight

        # set the dictionaries
        self.tag_dictionary: Dictionary = tag_dictionary
        # if we use a CRF, we must add special START and STOP tags to the dictionary
        if use_crf:
            raise NotImplementedError()

        self.tag_type: str = tag_type
        self.tagset_size: int = len(tag_dictionary)

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

        # initialize the network architecture
        self.module: dict = module.copy()
        self.nlayers: int = rnn_layers
        self.hidden_word = None

        # dropouts
        self.use_dropout: float = dropout
        self.use_word_dropout: float = word_dropout
        self.use_locked_dropout: float = locked_dropout

        self.pickle_module = pickle_module

        if dropout > 0.0:
            self.dropout = torch.nn.Dropout(dropout)

        if word_dropout > 0.0:
            self.word_dropout = flair.nn.WordDropout(word_dropout)

        if locked_dropout > 0.0:
            self.locked_dropout = flair.nn.LockedDropout(locked_dropout)

        embedding_dim: int = self.embeddings.embedding_length
        rnn_input_dim: int = embedding_dim

        # optional reprojection layer on top of word embeddings
        self.reproject_embeddings = reproject_embeddings
        if self.reproject_embeddings:
            if type(self.reproject_embeddings) == int:
                rnn_input_dim = self.reproject_embeddings

            self.embedding2nn = torch.nn.Linear(embedding_dim, rnn_input_dim)

        module_class = getattr(modules, module.pop("name"))
        self.mod = module_class(**module)

        self.train_initial_hidden_state = train_initial_hidden_state
        self.bidirectional = True
        self.rnn_type = rnn_type

        # bidirectional LSTM on top of embedding layer
        if self.use_rnn:
            num_directions = 2 if self.bidirectional else 1

            if self.rnn_type in ["LSTM", "GRU"]:

                self.rnn = getattr(torch.nn, self.rnn_type)(
                    rnn_input_dim,
                    hidden_size,
                    num_layers=self.nlayers,
                    dropout=0.0 if self.nlayers == 1 else 0.5,
                    bidirectional=True,
                    batch_first=True,
                )
                # Create initial hidden state and initialize it
                if self.train_initial_hidden_state:
                    self.hs_initializer = torch.nn.init.xavier_normal_

                    self.lstm_init_h = Parameter(
                        torch.randn(self.nlayers * num_directions, self.hidden_size),
                        requires_grad=True,
                    )

                    self.lstm_init_c = Parameter(
                        torch.randn(self.nlayers * num_directions, self.hidden_size),
                        requires_grad=True,
                    )

                    # TODO: Decide how to initialize the hidden state variables
                    # self.hs_initializer(self.lstm_init_h)
                    # self.hs_initializer(self.lstm_init_c)

            # final linear map to tag space
            self.linear = torch.nn.Linear(
                hidden_size * num_directions, len(tag_dictionary)
            )
        else:
            self.linear = torch.nn.Linear(
                rnn_input_dim, len(tag_dictionary)
            )

            if reuse_embedding_weight and isinstance(embeddings, StackedEmbeddings):
                for embedding in embeddings.embeddings:
                    if isinstance(embedding, OneHotEmbeddings):
                        self.linear.to(flair.device)
                        self.linear.weight.data = embedding.embedding_layer.weight[:-1]
                        log.info("Reuse embedding weights for output projection.")
                        break

        self.to(flair.device)
    
    def _get_state_dict(self):
        model_state = {
            "state_dict": self.state_dict(),
            "embeddings": self.embeddings,
            "hidden_size": self.hidden_size,
            "train_initial_hidden_state": self.train_initial_hidden_state,
            "tag_dictionary": self.tag_dictionary,
            "tag_type": self.tag_type,
            "use_crf": self.use_crf,
            "use_rnn": self.use_rnn,
            "rnn_layers": self.rnn_layers,
            "use_dropout": self.use_dropout,
            "use_word_dropout": self.use_word_dropout,
            "use_locked_dropout": self.use_locked_dropout,
            "rnn_type": self.rnn_type,
            "beta": self.beta,
            "weight_dict": self.weight_dict,
            "reproject_embeddings": self.reproject_embeddings,
            "reuse_embedding_weight": self.reuse_embedding_weight,
            "module": self.module,
        }
        return model_state

    @staticmethod
    def _init_model_with_state_dict(state):

        rnn_type = "LSTM" if "rnn_type" not in state.keys() else state["rnn_type"]
        use_dropout = 0.0 if "use_dropout" not in state.keys() else state["use_dropout"]
        use_word_dropout = (
            0.0 if "use_word_dropout" not in state.keys() else state["use_word_dropout"]
        )
        use_locked_dropout = (
            0.0
            if "use_locked_dropout" not in state.keys()
            else state["use_locked_dropout"]
        )
        train_initial_hidden_state = (
            False
            if "train_initial_hidden_state" not in state.keys()
            else state["train_initial_hidden_state"]
        )
        beta = 1.0 if "beta" not in state.keys() else state["beta"]
        weights = None if "weight_dict" not in state.keys() else state["weight_dict"]
        reproject_embeddings = True if "reproject_embeddings" not in state.keys() else state["reproject_embeddings"]
        if "reproject_to" in state.keys():
            reproject_embeddings = state["reproject_to"]
        reuse_embedding_weight = False if "reuse_embedding_weight" not in state.keys() else state['reuse_embedding_weight']

        model = MaskedLanguageModel(
            hidden_size=state["hidden_size"],
            embeddings=state["embeddings"],
            tag_dictionary=state["tag_dictionary"],
            tag_type=state["tag_type"],
            use_crf=state["use_crf"],
            use_rnn=state["use_rnn"],
            rnn_layers=state["rnn_layers"],
            dropout=use_dropout,
            word_dropout=use_word_dropout,
            locked_dropout=use_locked_dropout,
            train_initial_hidden_state=train_initial_hidden_state,
            rnn_type=rnn_type,
            beta=beta,
            loss_weights=weights,
            reproject_embeddings=reproject_embeddings,
            reuse_embedding_weight=reuse_embedding_weight,
            module=state["module"],
        )
        model.load_state_dict(state["state_dict"])
        return model
    
    def _calculate_loss(
            self, features: torch.tensor, sentences: List[Sentence]
    ) -> float:

        lengths: List[int] = [len(sentence.tokens) for sentence in sentences]

        pad_idx = len(self.tag_dictionary)+1

        tag_list: List = []
        for s_id, sentence in enumerate(sentences):
            # get the tags in this sentence
            tag_idx: List[int] = [
                self.tag_dictionary.get_idx_for_item(token.get_tag(self.tag_type).value) if token.get_tag(self.tag_type).value != '<pad>' else pad_idx
                for token in sentence
            ]
            # add tags as tensor
            tag = torch.tensor(tag_idx, device=flair.device)
            tag_list.append(tag)

        score = 0
        for sentence_feats, sentence_tags, sentence_length in zip(
                features, tag_list, lengths
        ):
            sentence_feats = sentence_feats[:sentence_length]
            score += torch.nn.functional.cross_entropy(
                sentence_feats, sentence_tags, weight=self.loss_weights, ignore_index=pad_idx
            )
        score /= len(features)

        return score
    
    def evaluate(
            self,
            sentences: Union[List[Sentence], Dataset],
            out_path: Union[str, Path] = None,
            embedding_storage_mode: str = "none",
            mini_batch_size: int = 32,
            num_workers: int = 8,
            wsd_evaluation: bool = False
    ) -> Tuple[Result, float]:

        # read Dataset into data loader (if list of sentences passed, make Dataset first)
        if not isinstance(sentences, Dataset):
            sentences = SentenceDataset(sentences)
        data_loader = DataLoader(sentences, batch_size=mini_batch_size, num_workers=num_workers)

        # else, use scikit-learn to evaluate
        y_true = []
        y_pred = []
        labels = Dictionary(add_unk=False)

        eval_loss = 0
        batch_no: int = 0

        lines: List[str] = []

        for batch in data_loader:

            # predict for batch
            loss = self.predict(batch,
                                embedding_storage_mode=embedding_storage_mode,
                                mini_batch_size=mini_batch_size,
                                label_name='predicted',
                                return_loss=True)
            eval_loss += loss * len(batch)
            batch_no += 1

            for sentence in batch:

                for token in sentence:
                    # add gold tag
                    gold_tag = token.get_tag(self.tag_type).value
                    y_true.append(labels.add_item(gold_tag))

                    # add predicted tag
                    if wsd_evaluation:
                        if gold_tag == 'O':
                            predicted_tag = 'O'
                        else:
                            predicted_tag = token.get_tag('predicted').value
                    else:
                        predicted_tag = token.get_tag('predicted').value

                    y_pred.append(labels.add_item(predicted_tag))

                    # for file output
                    lines.append(f'{token.text} {gold_tag} {predicted_tag}\n')

                lines.append('\n')

        if out_path:
            with open(Path(out_path), "w", encoding="utf-8") as outfile:
                outfile.write("".join(lines))

        eval_loss /= len(sentences)

        loss = eval_loss.item()

        # line for log file
        perplexity = 2**loss if loss < 100 else "over 2^100..."
        log_header = "PERPLEXITY"
        log_line = f"\t{str(perplexity)}"

        e_perplexity = math.exp(loss) if loss < 100 else "over e^100..."
        
        detailed_result = (
            "\nResults:"
            f"\n- Cross Entropy Loss    {str(loss)}"
            f"\n- Perplexity (2^loss)   {str(perplexity)}"
            f"\n- Perplexity (e^loss)   {str(e_perplexity)}"
            "\n"
        )

        result = Result(
            main_score=-loss,
            log_line=log_line,
            log_header=log_header,
            detailed_results=detailed_result,
        )
        return result, eval_loss