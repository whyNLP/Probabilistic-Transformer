import random
import os
import sys
import shutil
from pathlib import Path
from typing import Union, List
from collections import Counter

from torch.utils.data import Dataset, ConcatDataset, Subset

import flair
from flair.data import Corpus, Dictionary, Sentence, FlairDataset
from flair.models import LanguageModel
from flair.optim import *
from flair.training_utils import add_file_handler
from flair.file_utils import cached_path, unpack_file, unzip_file
from flair.datasets.base import find_train_dev_test_files

log = logging.getLogger("flair")


class PlainTextDataset(FlairDataset):
    def __init__(
            self,
            path: Union[str, Path],
            document_delimiter: str = '\n',
            encoding: str = "utf-8",
            skip_first_line: bool = False,
            in_memory: bool = True,
    ):
        if type(path) is str:
            path = Path(path)
        assert path.exists()

        self.path = path
        self.document_delimiter = document_delimiter
        # store either Sentence objects in memory, or only file offsets
        self.in_memory = in_memory

        self.total_sentence_count: int = 0

        # determine encoding of text file
        self.encoding = encoding

        with open(str(self.path), encoding=self.encoding) as file:

            # skip first line if to selected
            if skip_first_line:
                file.readline()

            # option 1: read only sentence boundaries as offset positions
            if not self.in_memory:

                raise NotImplementedError()

            # option 2: keep everything in memory
            if self.in_memory:

                lines = [doc.strip() for doc in file.read().split(self.document_delimiter) if doc.strip()]
                self.sentences: List[Sentence] = [Sentence([token.strip() for token in line.split() if token.strip()]) for line in lines]

                self.total_sentence_count = len(self.sentences)
    
    def is_in_memory(self) -> bool:
        return self.in_memory

    def __len__(self):
        return self.total_sentence_count

    def __getitem__(self, index: int = 0) -> Sentence:

        # if in memory, retrieve parsed sentence
        if self.in_memory:
            sentence = self.sentences[index]

        # else skip to position in file where sentence begins
        else:
            raise NotImplementedError()

        return sentence

class PlainTextCorpus(Corpus):
    def __init__(
            self,
            data_folder: Union[str, Path],
            train_file=None,
            test_file=None,
            dev_file=None,
            document_delimiter: str = '\n',
            encoding: str = "utf-8",
            skip_first_line: bool = False,
            in_memory: bool = True,
            autofind_splits: bool = True,
            min_freq: int = 1,
            **corpusargs,
    ):
        # find train, dev and test files if not specified
        dev_file, test_file, train_file = \
            find_train_dev_test_files(data_folder, dev_file, test_file, train_file, autofind_splits)

        # get train data
        train = PlainTextDataset(
            train_file,
            encoding=encoding,
            in_memory=in_memory,
            document_delimiter=document_delimiter,
            skip_first_line=skip_first_line,
        ) if train_file is not None else None

        # read in test file if exists
        test = PlainTextDataset(
            test_file,
            encoding=encoding,
            in_memory=in_memory,
            document_delimiter=document_delimiter,
            skip_first_line=skip_first_line,
        ) if test_file is not None else None

        # read in dev file if exists
        dev = PlainTextDataset(
            dev_file,
            encoding=encoding,
            in_memory=in_memory,
            document_delimiter=document_delimiter,
            skip_first_line=skip_first_line,
        ) if dev_file is not None else None

        self.min_freq = min_freq

        super(PlainTextCorpus, self).__init__(train, dev, test, name=str(data_folder), **corpusargs)
    
    def make_tag_dictionary(self, tag_type: str) -> Dictionary:

        if tag_type != 'mlm':
            log.warn(f"'{tag_type}' tag type used. Currently we only support MLM task for plain text corpus.")

        # Make the tag dictionary
        tag_dictionary: Dictionary = Dictionary()
        tokens = []
        for sentence in self.train:
            for token in sentence.tokens:
                tokens.append(token.text)
        most_common = Counter(tokens)
        tokens = sorted([token for token, freq in most_common.items() if freq >= self.min_freq])
        for token in tokens:
            tag_dictionary.add_item(token)
        return tag_dictionary


class StandardPTBCorpus(PlainTextCorpus):
    def __init__(
            self,
            base_path: Union[str, Path] = None,
            in_memory: bool = True,
            **corpusargs,
    ):
        """
        Initialize the standard 10,000 word Penn Treebank corpus. The first time you call this constructor it will 
        automatically download the dataset.
        :param base_path: Default is None, meaning that corpus gets auto-downloaded and loaded. You can override this
        to point to a different folder but typically this should not be necessary.
        :param in_memory: If True, keeps dataset in memory giving speedups in training.
        """
        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        std_ptb_path = "https://github.com/wojzaremba/lstm/raw/master/data/"

        # download files if not present locally
        cached_path(f"{std_ptb_path}ptb.test.txt", data_folder / 'raw')
        cached_path(f"{std_ptb_path}ptb.valid.txt", data_folder / 'raw')
        cached_path(f"{std_ptb_path}ptb.train.txt", data_folder / 'raw')

        # we need to slightly modify the original files by adding some new lines after document separators
        train_data_file = data_folder / 'train.txt'
        if not train_data_file.is_file():
            shutil.copyfile(str(data_folder / 'raw' / 'ptb.test.txt'), str(data_folder / 'test.txt'))
            shutil.copyfile(str(data_folder / 'raw' / 'ptb.valid.txt'), str(data_folder / 'dev.txt'))
            shutil.copyfile(str(data_folder / 'raw' / 'ptb.train.txt'), str(data_folder / 'train.txt'))

        super(StandardPTBCorpus, self).__init__(
            data_folder,
            train_file='train.txt',
            dev_file='dev.txt',
            test_file='test.txt',
            document_delimiter='\n',
            encoding='utf-8',
            skip_first_line=False,
            in_memory=in_memory,
            min_freq=1,
            **corpusargs,
        )


class BLLIPTextCorpus(PlainTextCorpus):
    def __init__(
            self,
            base_path: Union[str, Path] = None,
            mode: str = 'XS',
            min_freq: int = 1,
            in_memory: bool = False,
            use_subtoken: bool = True,
            **corpusargs,
    ):
        from .treebanks import BLLIPCorpus
        from transformers import GPT2Tokenizer
        from flair.embeddings.legacy import _build_token_subwords_mapping_gpt2
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")

        class TokenizeDatasetWrapper(PlainTextDataset):
            def __init__(self, dataset: Dataset):
                self.dataset = dataset
                self.total_sentence_count = len(dataset)

            def __getitem__(self, index: int = 0):
                sentence: Sentence = self.dataset[index]
                _, tokenized_string = _build_token_subwords_mapping_gpt2(sentence, tokenizer)
                subwords = tokenizer.tokenize(tokenized_string)
                return Sentence(subwords)
            
            def is_in_memory(self) -> bool:

                flair_dataset = self.dataset
                while True:
                    if type(flair_dataset) is Subset:
                        flair_dataset = flair_dataset.dataset
                    elif type(flair_dataset) is ConcatDataset:
                        flair_dataset = flair_dataset.datasets[0]
                    else:
                        break

                if type(flair_dataset) is list:
                    return True
                elif isinstance(flair_dataset, FlairDataset) and flair_dataset.is_in_memory():
                    return True
                return False

        # this dataset name
        dataset_name = self.__class__.__name__.lower()
        
        pos_corpus = BLLIPCorpus(base_path, mode, in_memory=in_memory)
        train = TokenizeDatasetWrapper(pos_corpus.train) if use_subtoken else pos_corpus.train
        dev = TokenizeDatasetWrapper(pos_corpus.dev) if use_subtoken else pos_corpus.dev
        test = TokenizeDatasetWrapper(pos_corpus.test) if use_subtoken else pos_corpus.test

        self.min_freq = min_freq

        super(PlainTextCorpus, self).__init__(train, dev, test, name=dataset_name, **corpusargs)