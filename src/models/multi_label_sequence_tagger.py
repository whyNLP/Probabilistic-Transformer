import logging
import sys

from typing import List, Union, Optional, Dict, Tuple
from pathlib import Path

import numpy as np
import torch
import torch.nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data.dataset import Dataset
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


class AttentionDecoder(torch.nn.Module):
    def __init__(self, in_features: int, rank: int = None) -> None:
        super(AttentionDecoder, self).__init__()

        if rank is None:
            rank = in_features
        
        self.linear_1 = torch.nn.Linear(in_features, rank)
        self.linear_2 = torch.nn.Linear(in_features, rank)

    def forward(self, x: torch.Tensor):
        x1 = self.linear_1(x)
        x2 = self.linear_2(x)
        return torch.bmm(x1, x2.transpose(-1, -2))

class MultiLabelSequenceTagger(SequenceTagger):
    def __init__(
            self,
            embeddings: TokenEmbeddings,
            tag_dictionary: List[Tuple[str, Dictionary, str]],
            module: dict,
            tag_type: str = None,
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
            loss_tag_weights: Dict[str, float] = None,
    ):
        """
        Initializes a SequenceTagger with multiple labels
        :param hidden_size: number of hidden states in RNN
        :param embeddings: word embeddings used in tagger
        :param tag_dictionary: tuples of (tag_type, tag_dictionary, prediction_head)
          where tag_type: string identifier for tag type, 
                tag_dictionary: the dictionary of tags you want to predict.
                prediction_head: Options: 'classifier', 'relative', 'attention'
                    'relative', 'attention' are only designed for dependency relationships.
        :param tag_type: unused, reserved for APIs
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
        :param loss_tag_weights: Dictionary of {tag_type: loss_weight}.
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

        # set the dictionaries
        self.tag_dictionarys: List[Tuple[str, Dictionary, str]] = tag_dictionary
        # if we use a CRF, we must add special START and STOP tags to the dictionary
        if use_crf:
            raise NotImplementedError
            for _, dictionary, _ in self.tag_dictionarys:
                dictionary.add_item(START_TAG)
                dictionary.add_item(STOP_TAG)

        self.beta = beta

        self.weight_dict = loss_weights
        # Initialize the weight tensor
        if loss_weights is not None:
            raise NotImplementedError
            n_classes = len(self.tag_dictionary)
            weight_list = [1. for i in range(n_classes)]
            for i, tag in enumerate(self.tag_dictionary.get_items()):
                if tag in loss_weights.keys():
                    weight_list[i] = loss_weights[tag]
            self.loss_weights = torch.FloatTensor(weight_list).to(flair.device)
        else:
            self.loss_weights = None
        
        self.loss_tag_weights = loss_tag_weights

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
        
        linear_modules = []
        for _, tag_dictionary, predict_head in self.tag_dictionarys:
            if predict_head in ('classifier', 'relative'):
                linear_modules.append(torch.nn.Linear(output_dim, len(tag_dictionary)))
            elif predict_head == 'attention':
                linear_modules.append(AttentionDecoder(output_dim))

        self.linear = torch.nn.ModuleList(linear_modules)

        if self.use_crf:
            raise NotImplementedError
            self.transitions = torch.nn.Parameter(
                torch.randn(self.tagset_size, self.tagset_size)
            )

            self.transitions.detach()[
            self.tag_dictionary.get_idx_for_item(START_TAG), :
            ] = -10000

            self.transitions.detach()[
            :, self.tag_dictionary.get_idx_for_item(STOP_TAG)
            ] = -10000

        self.to(flair.device)

    def _get_state_dict(self):
        model_state = {
            "state_dict": self.state_dict(),
            "embeddings": self.embeddings,
            "hidden_size": self.hidden_size,
            "train_initial_hidden_state": self.train_initial_hidden_state,
            "tag_dictionarys": self.tag_dictionarys,
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

        model = MultiLabelSequenceTagger(
            hidden_size=state["hidden_size"],
            embeddings=state["embeddings"],
            tag_dictionary=state["tag_dictionarys"],
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
        )
        model.load_state_dict(state["state_dict"])
        return model

    def forward(self, sentences: List[Sentence]) -> List[torch.Tensor]:

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

        features = [linear(sentence_tensor) for linear in self.linear]

        return features

    def _calculate_loss(
            self, features: List[torch.tensor], sentences: List[Sentence]
    ) -> float:

        lengths: List[int] = [len(sentence.tokens) for sentence in sentences]

        scores = 0
        for i, (tag_type, tag_dictionary, prediction_head) in enumerate(self.tag_dictionarys):

            if self.loss_tag_weights:
                loss_weight = self.loss_tag_weights.get(tag_type, 1.0)
            else:
                loss_weight = 1.0

            tag_list: List = []
            for s_id, sentence in enumerate(sentences):
                # get the tags in this sentence
                tag_idx: List[int] = [
                    tag_dictionary.get_idx_for_item(token.get_tag(tag_type).value)
                    for token in sentence
                ]
                # add tags as tensor
                tag = torch.tensor(tag_idx, device=flair.device)
                tag_list.append(tag)

            score = 0
            for sentence_feats, sentence_tags, sentence_length in zip(
                    features[i], tag_list, lengths
            ):
                # print(tag_type)
                # print(features.shape)
                # print(sentence_feats.shape)
                # print(sentence_tags.shape)
                sentence_feats = sentence_feats[:sentence_length]
                score += torch.nn.functional.cross_entropy(
                    sentence_feats, sentence_tags, weight=self.loss_weights
                )
            score /= len(features[i])

            scores += loss_weight * score

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

        from collections import defaultdict

        # read Dataset into data loader (if list of sentences passed, make Dataset first)
        if not isinstance(sentences, Dataset):
            sentences = SentenceDataset(sentences)
        data_loader = DataLoader(sentences, batch_size=mini_batch_size, num_workers=num_workers)

        # else, use scikit-learn to evaluate
        y_true = defaultdict(list)
        y_pred = defaultdict(list)
        labels = defaultdict(lambda: Dictionary(add_unk=False))

        eval_loss = 0
        batch_no: int = 0

        lines: List[str] = []

        correct_count = 0

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

                correct = True

                for token in sentence:

                    line = f'{token.text}'

                    for tag_type, _, _ in self.tag_dictionarys:
                        
                        # add gold tag
                        gold_tag = token.get_tag(tag_type).value
                        y_true[tag_type].append(labels[tag_type].add_item(gold_tag))

                        # add predicted tag
                        if wsd_evaluation:
                            if gold_tag == 'O':
                                predicted_tag = 'O'
                            else:
                                predicted_tag = token.get_tag(f'predicted-{tag_type}').value
                        else:
                            predicted_tag = token.get_tag(f'predicted-{tag_type}').value

                        y_pred[tag_type].append(labels[tag_type].add_item(predicted_tag))

                        line += f' {tag_type} {gold_tag} {predicted_tag}'

                        if gold_tag != predicted_tag:
                            correct = False

                    # for file output
                    lines.append(f'{line}\n')

                if correct:
                    correct_count += 1

                lines.append('\n')

        if out_path:
            with open(Path(out_path), "w", encoding="utf-8") as outfile:
                outfile.write("".join(lines))

        eval_loss /= batch_no

        # use sklearn
        from sklearn import metrics

        detailed_result = ""

        for tag_type, _, _ in self.tag_dictionarys:

            # make "classification report"
            target_names = []
            labels_to_report = []
            all_labels = []
            all_indices = []
            for i in range(len(labels[tag_type])):
                label = labels[tag_type].get_item_for_index(i)
                all_labels.append(label)
                all_indices.append(i)
                if label == '_' or label == '': continue
                target_names.append(label)
                labels_to_report.append(i)

            # report over all in case there are no labels
            if not labels_to_report:
                target_names = all_labels
                labels_to_report = all_indices

            classification_report = metrics.classification_report(y_true[tag_type], y_pred[tag_type], digits=4, target_names=target_names,
                                                                zero_division=1, labels=labels_to_report)

            # get scores
            micro_f_score = round(
                metrics.fbeta_score(y_true[tag_type], y_pred[tag_type], beta=self.beta, average='micro', labels=labels_to_report), 4)
            macro_f_score = round(
                metrics.fbeta_score(y_true[tag_type], y_pred[tag_type], beta=self.beta, average='macro', labels=labels_to_report), 4)
            accuracy_score = round(metrics.accuracy_score(y_true[tag_type], y_pred[tag_type]), 4)

            detailed_result += (
                    "\nResults:"
                    f"\n- F-score (micro): {micro_f_score}"
                    f"\n- F-score (macro): {macro_f_score}"
                    f"\n- Accuracy (incl. no class): {accuracy_score}"
                    '\n\nBy class:\n' + classification_report
            )

            # line for log file
            log_header = "ACCURACY"
            log_line = f"\t{accuracy_score}"
        
        accuracy_score = correct_count / len(sentences)
        log_line = f"\t{accuracy_score}"

        detailed_result += "OVERALL ACCURACY: {}\n".format(accuracy_score)

        result = Result(
            main_score=accuracy_score,
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

                for i, (tag_type, tag_dictionary, _) in enumerate(self.tag_dictionarys):

                    tags, all_tags = self._obtain_labels(
                        feature=feature[i],
                        tag_dictionary=tag_dictionary,
                        batch_sentences=batch,
                        transitions=transitions,
                        get_all_tags=all_tag_prob,
                    )

                    for (sentence, sent_tags) in zip(batch, tags):
                        for (token, tag) in zip(sentence.tokens, sent_tags):
                            token.add_tag_label(f"{label_name}-{tag_type}", tag)

                    # all_tags will be empty if all_tag_prob is set to False, so the for loop will be avoided
                    for (sentence, sent_all_tags) in zip(batch, all_tags):
                        for (token, token_all_tags) in zip(sentence.tokens, sent_all_tags):
                            token.add_tags_proba_dist(f"{label_name}-{tag_type}", token_all_tags)

                # clearing token embeddings to save memory
                store_embeddings(batch, storage_mode=embedding_storage_mode)

            if return_loss:
                return overall_loss / batch_no

    def _obtain_labels(
            self,
            feature: torch.Tensor,
            tag_dictionary: Dictionary,
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
        feature = feature.cpu()
        if self.use_crf:
            feature = feature.numpy()
        else:
            for index, length in enumerate(lengths):
                feature[index, length:] = 0
            softmax_batch = F.softmax(feature, dim=2).cpu()
            scores_batch, prediction_batch = torch.max(softmax_batch, dim=2)
            feature = zip(softmax_batch, scores_batch, prediction_batch)

        for feats, length in zip(feature, lengths):
            if self.use_crf:
                confidences, tag_seq, scores = self._viterbi_decode(
                    feats=feats[:length],
                    transitions=transitions,
                    all_scores=get_all_tags,
                )
            else:
                softmax, score, prediction = feats
                confidences = score[:length].tolist()
                tag_seq = prediction[:length].tolist()
                scores = softmax[:length].tolist()

            tags.append(
                [
                    (
                        Label(tag_dictionary.get_item_for_index(tag), conf)
                        if tag < len(tag_dictionary)
                        else Label("<out-of-dist>", conf)
                    )
                    for conf, tag in zip(confidences, tag_seq)
                ]
            )

            if get_all_tags:
                all_tags.append(
                    [
                        [
                            Label(
                                tag_dictionary.get_item_for_index(score_id), score
                            )
                            for score_id, score in enumerate(score_dist)
                        ]
                        for score_dist in scores
                    ]
                )

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