import random
import sys
import shutil
from pathlib import Path
from typing import Union, List

from torch.utils.data import Dataset

try:
    from apex import amp
except ImportError:
    amp = None

import flair
from flair.data import Corpus, Dictionary, Sentence
from flair.models import LanguageModel
from flair.optim import *
from flair.training_utils import add_file_handler
from flair.file_utils import cached_path, unpack_file, unzip_file
from flair.datasets.base import find_train_dev_test_files

log = logging.getLogger("flair")


class PlainTextDataset(Dataset):
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

        super(PlainTextCorpus, self).__init__(train, dev, test, name=str(data_folder), **corpusargs)


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
            **corpusargs,
        )
