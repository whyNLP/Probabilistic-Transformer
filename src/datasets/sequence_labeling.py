from os import rename
from pathlib import Path
import flair
from flair.datasets import CONLL_03
from flair.file_utils import cached_path, unpack_file

import json
import re
from typing import Any, List, Optional, Union, Tuple
from torch.utils.data import ConcatDataset, Dataset
from flair.data import Corpus, FlairDataset, Sentence, Dictionary
from flair.datasets.base import find_train_dev_test_files
from flair.file_utils import cached_path, unpack_file

class AUTO_CONLL_03(CONLL_03):
    def __init__(
            self,
            tag_to_bioes: str = "ner",
            in_memory: bool = True,
            **corpusargs,
    ):
        """
        Initialize the CoNLL-03 corpus. Auto-download supported.
        
        :param tag_to_bioes: NER by default, need not be changed, but you could also select 'pos' or 'np' to predict
        POS tags or chunks respectively
        :param in_memory: If True, keeps dataset in memory giving speedups in training.
        :param document_as_sequence: If True, all sentences of a document are read into a single Sentence object
        """
        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        if not (data_folder / "dev.txt").exists():
            # download and extract data
            conll03_path = cached_path('https://data.deepai.org/conll2003.zip', Path("datasets") / dataset_name)
            unpack_file(conll03_path, data_folder, "zip", True)
            rename(data_folder / "valid.txt", data_folder / "dev.txt")
            

        super(AUTO_CONLL_03, self).__init__(
            base_path=base_path,
            tag_to_bioes=tag_to_bioes,
            in_memory=in_memory,
            **corpusargs,
        )


class MultiFileJsonlCorpus(Corpus):
    """
    This class represents a generic Jsonl corpus with multiple train, dev, and test files.
    """

    def __init__(
        self,
        train_files=None,
        test_files=None,
        dev_files=None,
        encoding: str = "utf-8",
        text_column_name: str = "data",
        label_name_mapping: dict = None,
        label_content_mapping: dict = None,
        **corpusargs,
    ):
        """
        Instantiates a MuliFileJsonlCorpus as, e.g., created with doccanos JSONL export.
        Note that at least one of train_files, test_files, and dev_files must contain one path.
        Otherwise, the initialization will fail.
        :param corpusargs: Additional arguments for Corpus initialization
        :param train_files: the name of the train files
        :param test_files: the name of the test files
        :param dev_files: the name of the dev files, if empty, dev data is sampled from train
        :param text_column_name: Name of the text column inside the jsonl files.
        :param label_name_mapping: A mapping of labels in jsonl and loaded dataset.
        :param label_content_mapping: A mapping of the label content. Should be
                Dict[label_column_name, Function[label, index -> new_label]]
        :raises RuntimeError: If no paths are given
        """
        train: Optional[Dataset] = (
            ConcatDataset(
                [
                    JsonlDataset(
                        train_file,
                        text_column_name=text_column_name,
                        label_name_mapping=label_name_mapping,
                        label_content_mapping=label_content_mapping,
                        encoding=encoding,
                    )
                    for train_file in train_files
                ]
            )
            if train_files and train_files[0]
            else None
        )

        # read in test file if exists
        test: Optional[Dataset] = (
            ConcatDataset(
                [
                    JsonlDataset(
                        test_file,
                        text_column_name=text_column_name,
                        label_name_mapping=label_name_mapping,
                        label_content_mapping=label_content_mapping,
                        encoding=encoding,
                    )
                    for test_file in test_files
                ]
            )
            if test_files and test_files[0]
            else None
        )

        # read in dev file if exists
        dev: Optional[Dataset] = (
            ConcatDataset(
                [
                    JsonlDataset(
                        dev_file,
                        text_column_name=text_column_name,
                        label_name_mapping=label_name_mapping,
                        label_content_mapping=label_content_mapping,
                        encoding=encoding,
                    )
                    for dev_file in dev_files
                ]
            )
            if dev_files and dev_files[0]
            else None
        )
        super().__init__(train, dev, test, **corpusargs)


class JsonlCorpus(MultiFileJsonlCorpus):
    def __init__(
        self,
        data_folder: Union[str, Path],
        train_file: Optional[Union[str, Path]] = None,
        test_file: Optional[Union[str, Path]] = None,
        dev_file: Optional[Union[str, Path]] = None,
        encoding: str = "utf-8",
        text_column_name: str = "data",
        label_name_mapping: dict = None,
        label_content_mapping: dict = None,
        autofind_splits: bool = True,
        name: Optional[str] = None,
        **corpusargs,
    ):
        """
        Instantiates a JsonlCorpus with one file per Dataset (train, dev, and test).
        :param data_folder: Path to the folder containing the JSONL corpus
        :param train_file: the name of the train file
        :param test_file: the name of the test file
        :param dev_file: the name of the dev file, if None, dev data is sampled from train
        :param text_column_name: Name of the text column inside the JSONL file.
        :param label_name_mapping: A mapping of labels in jsonl and loaded dataset.
        :param label_content_mapping: A mapping of the label content. Should be
                Dict[label_column_name, Function[label, index -> new_label]]
        :param autofind_splits: Whether train, test and dev file should be determined automatically
        :param name: name of the Corpus see flair.data.Corpus
        """
        # find train, dev and test files if not specified
        dev_file, test_file, train_file = find_train_dev_test_files(
            data_folder, dev_file, test_file, train_file, autofind_splits
        )
        super().__init__(
            dev_files=[dev_file] if dev_file else [],
            train_files=[train_file] if train_file else [],
            test_files=[test_file] if test_file else [],
            text_column_name=text_column_name,
            label_name_mapping=label_name_mapping,
            name=name if data_folder is None else str(data_folder),
            encoding=encoding,
            label_content_mapping=label_content_mapping,
            **corpusargs,
        )


class JsonlDataset(FlairDataset):
    def __init__(
        self,
        path_to_jsonl_file: Union[str, Path],
        encoding: str = "utf-8",
        text_column_name: str = "data",
        label_name_mapping: dict = None,
        label_content_mapping: dict = None
    ):
        """
        Instantiates a JsonlDataset and converts all annotated char spans to token tags using the IOB scheme.
        The expected file format is:
        { "<text_column_name>": "[<text_1>,...,<text_n>]", "label_column_name1": [<label_1>,...,<label_n>], "label_column_name2": [<label_1>,...,<label_n>] }
        :param path_to_jsonl_file: File to read
        :param text_column_name: Name of the text column
        :param label_name_mapping: A mapping of labels in jsonl and loaded dataset.
        :param label_content_mapping: A mapping of the label content. Should be
                Dict[label_column_name, Function[label, index -> new_label]]
        """
        path_to_json_file = Path(path_to_jsonl_file)

        self.text_column_name = text_column_name
        self.label_name_mapping = label_name_mapping
        self.label_content_mapping = label_content_mapping
        self.path_to_json_file = path_to_json_file

        self.sentences: List[Sentence] = []
        with path_to_json_file.open(encoding=encoding) as jsonl_fp:
            for line in jsonl_fp:
                current_line = json.loads(line)
                current_sentence = self._create_sentence_from_jsonl(current_line)
                self.sentences.append(current_sentence)

    def _create_sentence_from_jsonl(self, jsonl: dict) -> Sentence:
        """
        Create sentence from the json line.
        """
        raw_text = jsonl[self.text_column_name]
        current_labels = list(jsonl.keys())
        current_labels.remove(self.text_column_name)
        current_sentence = Sentence(raw_text)

        for label_name in current_labels:
            raw_labels = jsonl[label_name]

            if self.label_name_mapping:
                if label_name in self.label_name_mapping:
                    label_name = self.label_name_mapping[label_name]
                else:
                    continue
            
            for idx, (token, label) in enumerate(zip(current_sentence, raw_labels)):
                if self.label_content_mapping and label_name in self.label_content_mapping:
                    mapping = self.label_content_mapping[label_name]
                    label = mapping(label, idx)
                token.add_label(label_name, label)
        
        return current_sentence

    def is_in_memory(self) -> bool:
        """
        Currently all Jsonl Datasets are stored in Memory
        """
        return True

    def __len__(self):
        """
        Number of sentences in the Dataset
        """
        return len(self.sentences)

    def __getitem__(self, index: int = 0) -> Sentence:
        """
        Returns the sentence at a given index
        """
        return self.sentences[index]


class COGS_SequenceLabeling(JsonlCorpus):
    def __init__(self, base_path: Union[str, Path] = None, in_memory: bool = True, dev_split: str = 'dev', test_split: str = 'test', mapping: str = 'classifier'):
        """
        :param dev_split: Options: 'dev', 'gen'
        :param test_split: Options: 'test', 'gen'
        :param mapping: Options: 'classifier', 'relative', 'attention'
        """
        self.mapping = mapping
        assert in_memory, "COGS_SequenceLabeling only supports in_memory setting."

        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        if dev_split == 'dev':
            dev_set = "dev_seqtag.jsonl"
        elif dev_split == 'gen':
            dev_set = "gen_dev_seqtag.jsonl"
        else:
            raise ValueError(f"Unknown test split: {dev_split}. Options: 'dev', 'gen'")

        if test_split == 'test':
            test_set = "test_seqtag.jsonl"
        elif test_split == 'gen':
            test_set = "gen_seqtag.jsonl"
        else:
            raise ValueError(f"Unknown test split: {test_split}. Options: 'test', 'gen'")

        # download data if necessary
        web_path = "https://raw.githubusercontent.com/bergen/EdgeTransformer/main/cogs/data"
        cached_path(f"{web_path}/{dev_set}", Path("datasets") / dataset_name)
        cached_path(
            f"{web_path}/{test_set}", Path("datasets") / dataset_name
        )
        cached_path(
            f"{web_path}/train_seqtag.jsonl", Path("datasets") / dataset_name
        )

        def mapping_classifier(label, idx):
            return str(label)

        def mapping_relative(label, idx):
            if int(label) == -1:
                return str(-1)
            else:
                return str(int(label) - idx)
        
        def mapping_attention(label, idx):
            if int(label) == -1:
                return str(idx)
            else:
                return str(label)

        if mapping == 'classifier':
            label_content_mapping = {"parent": mapping_classifier}
        elif mapping == 'relative':
            label_content_mapping = {"parent": mapping_relative}
        elif mapping == 'attention':
            label_content_mapping = {"parent": mapping_attention}

        super(COGS_SequenceLabeling, self).__init__(data_folder, label_content_mapping=label_content_mapping, text_column_name="tokens",
        train_file='train_seqtag.jsonl', dev_file=dev_set, test_file=test_set)

    def make_tag_dictionary(self, tag_type: str) -> List[Tuple[str, Dictionary, str]]:

        tag_dictionarys = []
        for tag_type in ('role', 'category', 'noun_type', 'verb_name'):
            # Make the tag dictionary
            tag_dictionary: Dictionary = Dictionary(add_unk=False)
            for sentence in self.get_all_sentences():
                for token in sentence.tokens:
                    tag_dictionary.add_item(token.get_tag(tag_type).value)
            tag_dictionarys.append((tag_type, tag_dictionary, 'classifier'))

        tag_type = 'parent'
        if self.mapping in ('classifier', 'relative'):
            tag_dictionary: Dictionary = Dictionary(add_unk=False)
            for sentence in self.get_all_sentences():
                for token in sentence.tokens:
                    tag_dictionary.add_item(token.get_tag(tag_type).value)
        elif self.mapping == 'attention':
            tag_dictionary: Dictionary = Dictionary(add_unk=False)
            max_length = max(len(sentence) for sentence in self.get_all_sentences())
            for i in range(max_length):
                tag_dictionary.add_item(str(i))
        tag_dictionarys.append((tag_type, tag_dictionary, self.mapping))

        return tag_dictionarys