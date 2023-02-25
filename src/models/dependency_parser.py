import logging
import sys

from typing import List, Union, Optional, Dict, Tuple
from pathlib import Path

import numpy as np
import torch
import torch.nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from torch.nn.parameter import Parameter
from tqdm import tqdm

import flair.nn
from flair.data import Dictionary, Sentence, Label
from flair.embeddings import TokenEmbeddings, StackedEmbeddings, OneHotEmbeddings
from flair.models import SequenceTagger

from flair.datasets import SentenceDataset, DataLoader
from flair.training_utils import Result, store_embeddings

from . import modules

log = logging.getLogger("flair")

START_TAG: str = "<START>"
STOP_TAG: str = "<STOP>"


CONTENT_DEPRELS = {
    "NSUBJ", "OBJ", "IOBJ", "CSUBJ", "CCOMP", "XCOMP", "OBL", "VOCATIVE",
    "EXPL", "DISLOCATED", "ADVCL", "ADVMOD", "DISCOURSE", "NMOD", "APPOS",
    "NUMMOD", "ACL", "AMOD", "CONJ", "FIXED", "FLAT", "COMPOUND", "LIST",
    "PARATAXIS", "ORPHAN", "GOESWITH", "REPARANDUM", "ROOT", "DEP"
}

## Codes modified from source code of stanza:
##   https://github.com/stanfordnlp/stanza
class PairwiseBilinear(torch.nn.Module):
    ''' A bilinear module that deals with broadcasting for efficient memory usage.
    Input: tensors of sizes (N x L1 x D1) and (N x L2 x D2)
    Output: tensor of size (N x L1 x L2 x O)'''
    def __init__(self, input1_size, input2_size, output_size, bias=True):
        super().__init__()

        self.input1_size = input1_size
        self.input2_size = input2_size
        self.output_size = output_size

        self.weight = torch.nn.Parameter(torch.Tensor(input1_size, input2_size, output_size))
        self.bias = torch.nn.Parameter(torch.Tensor(output_size)) if bias else 0

    def forward(self, input1, input2):
        input1_size = list(input1.size())
        input2_size = list(input2.size())
        output_size = [input1_size[0], input1_size[1], input2_size[1], self.output_size]

        # ((N x L1) x D1) * (D1 x (D2 x O)) -> (N x L1) x (D2 x O)
        intermediate = torch.mm(input1.view(-1, input1_size[-1]), self.weight.view(-1, self.input2_size * self.output_size))
        # (N x L2 x D2) -> (N x D2 x L2)
        input2 = input2.transpose(1, 2)
        # (N x (L1 x O) x D2) * (N x D2 x L2) -> (N x (L1 x O) x L2)
        output = intermediate.view(input1_size[0], input1_size[1] * self.output_size, input2_size[2]).bmm(input2)
        # (N x (L1 x O) x L2) -> (N x L1 x L2 x O)
        output = output.view(input1_size[0], input1_size[1], self.output_size, input2_size[1]).transpose(2, 3)

        return output

class PairwiseBiaffineScorer(torch.nn.Module):
    def __init__(self, input1_size, input2_size, output_size):
        super().__init__()
        self.W_bilin = PairwiseBilinear(input1_size + 1, input2_size + 1, output_size)

        self.W_bilin.weight.data.zero_()
        self.W_bilin.bias.data.zero_()

    def forward(self, input1, input2):
        input1 = torch.cat([input1, input1.new_ones(*input1.size()[:-1], 1)], len(input1.size())-1)
        input2 = torch.cat([input2, input2.new_ones(*input2.size()[:-1], 1)], len(input2.size())-1)
        return self.W_bilin(input1, input2)

class DeepBiaffineScorer(torch.nn.Module):
    def __init__(self, input1_size, input2_size, hidden_size, output_size, hidden_func=F.relu, dropout=0):
        super().__init__()
        self.W1 = torch.nn.Linear(input1_size, hidden_size)
        self.W2 = torch.nn.Linear(input2_size, hidden_size)
        self.hidden_func = hidden_func
        self.scorer = PairwiseBiaffineScorer(hidden_size, hidden_size, output_size)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        input1 = input2 = x
        return self.scorer(self.dropout(self.hidden_func(self.W1(input1))), self.dropout(self.hidden_func(self.W2(input2))))


class DependencyParser(SequenceTagger):
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
            train_initial_hidden_state: bool = False,
            rnn_type: str = "LSTM",
            pickle_module: str = "pickle",
            beta: float = 1.0,
            loss_weights: Dict[str, float] = None,
            use_nonprojective: bool = True
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
        :param use_nonprojective: Use non-projective algorithm for tree decoding.

        """

        super(SequenceTagger, self).__init__()
        self.use_rnn = use_rnn
        self.hidden_size = hidden_size
        self.use_crf: bool = use_crf
        self.rnn_layers: int = rnn_layers

        self.trained_epochs: int = 0

        self.embeddings = embeddings
        self.use_nonprojective = use_nonprojective

        # set the dictionaries
        self.tag_dictionary: Dictionary = tag_dictionary
        # if we use a CRF, we must add special START and STOP tags to the dictionary
        if use_crf:
            self.tag_dictionary.add_item(START_TAG)
            self.tag_dictionary.add_item(STOP_TAG)

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
            output_dim = hidden_size * num_directions
        else:
            output_dim = rnn_input_dim

        if self.use_crf:
            self.transitions = torch.nn.Parameter(
                torch.randn(self.tagset_size, self.tagset_size)
            )

            self.transitions.detach()[
            self.tag_dictionary.get_idx_for_item(START_TAG), :
            ] = -10000

            self.transitions.detach()[
            :, self.tag_dictionary.get_idx_for_item(STOP_TAG)
            ] = -10000

        self.cls_embedding = torch.nn.Parameter(torch.randn(1, self.embeddings.embedding_length))

        self.dep_head_scorer = DeepBiaffineScorer(output_dim, output_dim, hidden_size, 1, dropout=dropout)
        self.dep_label_scorer = DeepBiaffineScorer(output_dim, output_dim, hidden_size, len(tag_dictionary), dropout=dropout)

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
            "module": self.module,
            "use_nonprojective": self.use_nonprojective
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
        use_nonprojective = True if "use_nonprojective" not in state.keys() else state["use_nonprojective"]

        model = DependencyParser(
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
            module=state["module"],
            use_nonprojective=use_nonprojective
        )
        model.load_state_dict(state["state_dict"])
        return model

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

        # --------------------------------------------------------------------
        # FF PART
        # --------------------------------------------------------------------
        if self.use_dropout > 0.0:
            sentence_tensor = self.dropout(sentence_tensor)
        if self.use_word_dropout > 0.0:
            sentence_tensor = self.word_dropout(sentence_tensor)
        if self.use_locked_dropout > 0.0:
            sentence_tensor = self.locked_dropout(sentence_tensor)

        if self.reproject_embeddings:
            sentence_tensor = self.embedding2nn(sentence_tensor)

        ## Add root node
        sentence_tensor = torch.cat((sentence_tensor, self.cls_embedding.repeat_interleave(len(sentences), dim=0).unsqueeze(1)), dim=1)
        mask = torch.cat((mask, torch.ones(len(sentences), 1, device=flair.device)), dim=1)
        
        sentence_tensor = self.mod(sentence_tensor, mask)

        if self.use_rnn:
            packed = torch.nn.utils.rnn.pack_padded_sequence(
                sentence_tensor, lengths, enforce_sorted=False, batch_first=True
            )

            # if initial hidden state is trainable, use this state
            if self.train_initial_hidden_state:
                initial_hidden_state = [
                    self.lstm_init_h.unsqueeze(1).repeat(1, len(sentences), 1),
                    self.lstm_init_c.unsqueeze(1).repeat(1, len(sentences), 1),
                ]
                rnn_output, hidden = self.rnn(packed, initial_hidden_state)
            else:
                rnn_output, hidden = self.rnn(packed)

            sentence_tensor, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(
                rnn_output, batch_first=True
            )

            if self.use_dropout > 0.0:
                sentence_tensor = self.dropout(sentence_tensor)
            # word dropout only before LSTM - TODO: more experimentation needed
            # if self.use_word_dropout > 0.0:
            #     sentence_tensor = self.word_dropout(sentence_tensor)
            if self.use_locked_dropout > 0.0:
                sentence_tensor = self.locked_dropout(sentence_tensor)

        dep_head_score = self.dep_head_scorer(sentence_tensor).squeeze(-1)
        dep_label_score = self.dep_label_scorer(sentence_tensor)

        diag = torch.eye(longest_token_sequence_in_batch+1, dtype=torch.bool, device=flair.device).unsqueeze(0)
        dep_head_score.masked_fill_(diag, -1e9)
        features = [dep_head_score, dep_label_score]

        return features

    def _calculate_loss(
            self, features: List[torch.tensor], sentences: List[Sentence]
    ) -> float:

        lengths: List[int] = [len(sentence.tokens) for sentence in sentences]

        scores = 0

        tag_list: List = []
        for s_id, sentence in enumerate(sentences):
            # get the tags in this sentence
            tag_idx: List[int] = [
                self.tag_dictionary.get_idx_for_item(token.get_tag(self.tag_type).value)
                for token in sentence
            ]
            # add tags as tensor
            tag = torch.tensor(tag_idx, device=flair.device)
            tag_list.append(tag)
        
        dep_list: List = []
        for s_id, sentence in enumerate(sentences):
            # get the tags in this sentence
            head_idx: List[int] = [
                token.head_id
                for token in sentence
            ]
            # add tags as tensor
            tag = torch.tensor(head_idx, device=flair.device)
            dep_list.append(tag)

        scores = 0
        for head_feats, sentence_feats, head_tags, sentence_tags, sentence_length in zip(
                features[0], features[1], dep_list, tag_list, lengths
        ):
            sentence_feats = sentence_feats[1:sentence_length+1, :sentence_length+1]
            head_feats = head_feats[1:sentence_length+1, :sentence_length+1]
            scores += torch.nn.functional.cross_entropy(
                head_feats, head_tags, weight=self.loss_weights
            )

            sentence_feats = sentence_feats[torch.arange(sentence_length), head_tags]
            scores += torch.nn.functional.cross_entropy(
                sentence_feats, sentence_tags, weight=self.loss_weights
            )
        scores /= len(sentences)

        return scores

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
        
        correct_edges, total_edges = 0, 0
        correct_c_edges, total_c_edges = 0, 0
        correct_c_sents, correct_sents, total_sents = 0, 0, 0

        for batch in data_loader:

            # predict for batch
            loss = self.predict(batch,
                                embedding_storage_mode=embedding_storage_mode,
                                mini_batch_size=mini_batch_size,
                                label_name='predicted',
                                return_loss=True)
            eval_loss += loss
            batch_no += 1

            for sentence in batch:
                
                correct, correct_c = True, True

                for token in sentence:
                    # add gold tag
                    gold_tag = token.get_tag(self.tag_type).value
                    y_true.append(labels.add_item(gold_tag))

                    # add predicted tag
                    if wsd_evaluation:
                        if gold_tag == 'O':
                            predicted_tag = 'O'
                        else:
                            predicted_tag = token.get_tag('predicted-label').value
                    else:
                        predicted_tag = token.get_tag('predicted-label').value

                    y_pred.append(labels.add_item(predicted_tag))

                    gold_id = str(token.head_id)
                    predicted_id = str(token.get_tag('predicted-head').value)

                    if gold_tag in CONTENT_DEPRELS:
                        total_c_edges += 1
                        if gold_id == predicted_id and gold_tag == predicted_tag:
                            correct_c_edges += 1
                        else:
                            correct_c = False
                    
                    total_edges += 1
                    if gold_id == predicted_id and gold_tag == predicted_tag:
                        correct_edges += 1
                    else:
                        correct = False

                    # for file output
                    lines.append(f'{token.text} {gold_id} {predicted_id} {gold_tag} {predicted_tag}\n')

                lines.append('\n')

                total_sents += 1
                if correct_c:
                    correct_c_sents += 1
                if correct:
                    correct_sents += 1

        if out_path:
            with open(Path(out_path), "w", encoding="utf-8") as outfile:
                outfile.write("".join(lines))

        eval_loss /= batch_no

        # use sklearn
        from sklearn import metrics

        # make "classification report"
        target_names = []
        labels_to_report = []
        all_labels = []
        all_indices = []
        for i in range(len(labels)):
            label = labels.get_item_for_index(i)
            all_labels.append(label)
            all_indices.append(i)
            if label == '_' or label == '': continue
            target_names.append(label)
            labels_to_report.append(i)

        # report over all in case there are no labels
        if not labels_to_report:
            target_names = all_labels
            labels_to_report = all_indices

        classification_report = metrics.classification_report(y_true, y_pred, digits=4, target_names=target_names,
                                                              zero_division=1, labels=labels_to_report)

        # get scores
        micro_f_score = round(
            metrics.fbeta_score(y_true, y_pred, beta=self.beta, average='micro', labels=labels_to_report), 4)
        macro_f_score = round(
            metrics.fbeta_score(y_true, y_pred, beta=self.beta, average='macro', labels=labels_to_report), 4)
        accuracy_score = round(metrics.accuracy_score(y_true, y_pred), 4)

        detailed_result = (
                "\nResults:"
                f"\n- F-score (micro): {micro_f_score}"
                f"\n- F-score (macro): {macro_f_score}"
                f"\n- Accuracy (incl. no class): {accuracy_score}"
                '\n\nBy class:\n' + classification_report
        )

        # See details in https://aclanthology.org/2022.acl-long.448.pdf
        # - LAS: Labeled Attachment Score
        # - CLAS: Content-word Labeled Attachment Score
        # - EM: Exact Match
        # - WCLAS: Whole Sentence Content-word Labeled Attachment Score
        las = correct_edges / total_edges if total_edges else 0.
        clas = correct_c_edges / total_c_edges if total_c_edges else 0.
        em = correct_sents / total_sents if total_sents else 0.
        wclas = correct_c_sents / total_sents if total_sents else 0.

        detailed_result += (
                "\nSummary:"
                f"\n- LAS: {las}"
                f"\n- CLAS: {clas}"
                f"\n- EM: {em}"
                f"\n- WCLAS: {wclas}"
        )

        # line for log file
        log_header = "LAS CLAS EM WCLAS"
        log_line = f"\t{las}\t{clas}\t{em}\t{wclas}"

        result = Result(
            main_score=em,
            log_line=log_line,
            log_header=log_header,
            detailed_results=detailed_result,
        )
        return result, eval_loss

    def predict(
            self,
            sentences: Union[List[Sentence], Sentence],
            mini_batch_size=32,
            all_tag_prob: bool = False,
            verbose: bool = False,
            label_name: Optional[str] = None,
            return_loss=False,
            embedding_storage_mode="none",
    ):
        """
        Predict sequence tags for Named Entity Recognition task
        :param sentences: a Sentence or a List of Sentence
        :param mini_batch_size: size of the minibatch, usually bigger is more rapid but consume more memory,
        up to a point when it has no more effect.
        :param all_tag_prob: True to compute the score for each tag on each token,
        otherwise only the score of the best tag is returned
        :param verbose: set to True to display a progress bar
        :param return_loss: set to True to return loss
        :param label_name: set this to change the name of the label type that is predicted
        :param embedding_storage_mode: default is 'none' which is always best. Only set to 'cpu' or 'gpu' if
        you wish to not only predict, but also keep the generated embeddings in CPU or GPU memory respectively.
        'gpu' to store embeddings in GPU memory.
        """
        if label_name == None:
            label_name = self.tag_type

        with torch.no_grad():
            if not sentences:
                return sentences

            if isinstance(sentences, Sentence):
                sentences = [sentences]

            # set context if not set already
            previous_sentence = None
            for sentence in sentences:
                if sentence.is_context_set(): continue
                sentence._previous_sentence = previous_sentence
                sentence._next_sentence = None
                if previous_sentence: previous_sentence._next_sentence = sentence
                previous_sentence = sentence

            # reverse sort all sequences by their length
            rev_order_len_index = sorted(
                range(len(sentences)), key=lambda k: len(sentences[k]), reverse=True
            )

            reordered_sentences: List[Union[Sentence, str]] = [
                sentences[index] for index in rev_order_len_index
            ]

            dataloader = DataLoader(
                dataset=SentenceDataset(reordered_sentences), batch_size=mini_batch_size
            )

            if self.use_crf:
                transitions = self.transitions.detach().cpu().numpy()
            else:
                transitions = None

            # progress bar for verbosity
            if verbose:
                dataloader = tqdm(dataloader)

            overall_loss = 0
            batch_no = 0
            for batch in dataloader:

                batch_no += 1

                if verbose:
                    dataloader.set_description(f"Inferencing on batch {batch_no}")

                batch = self._filter_empty_sentences(batch)
                # stop if all sentences are empty
                if not batch:
                    continue

                feature = self.forward(batch)

                if return_loss:
                    overall_loss += self._calculate_loss(feature, batch)

                tags, all_tags = self._obtain_labels(
                    feature=feature,
                    batch_sentences=batch,
                    transitions=transitions,
                    get_all_tags=all_tag_prob,
                )

                for (sentence, sent_tags) in zip(batch, tags):
                    for (token, tag) in zip(sentence.tokens, sent_tags):
                        token.add_tag_label(label_name + '-head', tag[0])
                        token.add_tag_label(label_name + '-label', tag[1])

                # all_tags will be empty if all_tag_prob is set to False, so the for loop will be avoided
                for (sentence, sent_all_tags) in zip(batch, all_tags):
                    for (token, token_all_tags) in zip(sentence.tokens, sent_all_tags):
                        token.add_tags_proba_dist(label_name, token_all_tags)

                # clearing token embeddings to save memory
                store_embeddings(batch, storage_mode=embedding_storage_mode)

            if return_loss:
                return overall_loss / batch_no
    
    def _obtain_labels(
            self,
            feature: torch.Tensor,
            batch_sentences: List[Sentence],
            transitions: Optional[np.ndarray],
            get_all_tags: bool,
    ) -> Tuple[List[List[Label]], List[List[List[Label]]]]:
        """
        Returns a tuple of two lists:
         - The first list corresponds to the most likely `Label` per token in each sentence.
         - The second list contains a probability distribution over all `Labels` for each token
           in a sentence for all sentences.
        """

        lengths: List[int] = [len(sentence.tokens) for sentence in batch_sentences]

        tags = []
        all_tags = []
        head_feature, label_feature = feature
        head_feature = head_feature.cpu()
        label_feature = label_feature.cpu()
        
        for index, length in enumerate(lengths):
            head_feature[index, length+1:] = 0
            label_feature[index, length+1:] = 0

        if self.use_nonprojective:
            from .modules.recorder import nonprojective_parse
            softmax_batch = F.softmax(head_feature, dim=2).cpu()
            scores_batch, prediction_batch = [], []
            for idx, (heads, length) in enumerate(zip(head_feature, lengths)):
                heads[torch.arange(1, length+1), torch.arange(1, length+1)] = 0 # recover diag, otherwise nonprojective_parse will produce unexpected results
                indices = nonprojective_parse(heads[1:length+1, :length+1])
                confidence = [softmax_batch[idx, i+1, h].item() for i, h in enumerate(indices)]
                scores_batch.append(confidence)
                prediction_batch.append(indices)
        else:
            softmax_batch = F.softmax(head_feature, dim=2).cpu()
            scores_batch, prediction_batch = torch.max(softmax_batch, dim=2)
        head_feature = zip(softmax_batch, scores_batch, prediction_batch)

        softmax_batch = F.softmax(label_feature, dim=3).cpu()
        scores_batch, prediction_batch = torch.max(softmax_batch, dim=3)
        label_feature = zip(softmax_batch, scores_batch, prediction_batch)

        for head_feats, label_feats, length in zip(head_feature, label_feature, lengths):
            
            _, score, prediction = head_feats

            if self.use_nonprojective:
                head_confidences = score
                head_tag_seq = prediction
            else:
                head_confidences = score[1:length+1].tolist()
                head_tag_seq = prediction[1:length+1].tolist()

            _, score, prediction = label_feats
            score, prediction = score[1:length+1], prediction[1:length+1]
            label_confidences = score[torch.arange(length), head_tag_seq].tolist()
            label_tag_seq = prediction[torch.arange(length), head_tag_seq].tolist()

            tags.append(
                [
                    (
                        Label(str(head_tag), head_conf),
                        Label(self.tag_dictionary.get_item_for_index(label_tag), label_conf)
                    )
                    for head_conf, head_tag, label_conf, label_tag in zip(head_confidences, head_tag_seq, label_confidences, label_tag_seq)
                ]
            )

            if get_all_tags:
                raise NotImplementedError

        return tags, all_tags

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