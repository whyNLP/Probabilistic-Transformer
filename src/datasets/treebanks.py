import logging
from pathlib import Path
from typing import List, Union
from collections import Counter
import os

from torch.utils.data.dataset import ConcatDataset, Subset

import flair
from flair.data import (
    Sentence,
    Corpus,
    Dictionary,
    Token,
    FlairDataset,
    randomly_split_into_two_datasets
)
from flair.datasets import ColumnDataset
from flair.file_utils import cached_path, unpack_file

import nltk
from nltk.corpus.reader.bracket_parse import BracketParseCorpusReader

log = logging.getLogger("flair")

class PennTreebankCorpus(Corpus):
    def __init__(self, splits: str = "2-21|22|23", base_path: Union[str, Path] = None, in_memory: bool = True, **kwargs):
        """
        Instantiates a Corpus from original Penn Treebank 3 dataset.

        :param splits: Section splits for Penn Treebank.
            A split is composed of 3 datasets (train|dev|test), seperated by '|'. Each dataset 
            is represented by numbers seperated by commas. You can use '-' to represent a continuous
            range of numbers (e.g.: "0,1,3-5,9|7|22-24"). These numbers are the section numbers in 
            Penn Treebank, from 0 to 24. Default: "2-21|22|23"
        :param base_path: Base path for flair storage. Usually use default setting.
        :param in_memory: If set to True, keeps full dataset in memory, otherwise does disk reads
        :return: a Corpus with annotated train, dev and test data
        """
        self.__dict__.update(kwargs)

        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        web_path = "https://github.com/shrutirij/deeplearning-project/raw/master/data/PennTreebank/treebank_3_LDC99T42.tgz"
        # web_path = "https://raw.githubusercontents.com/shrutirij/deeplearning-project/master/data/PennTreebank/treebank_3_LDC99T42.tgz" # use proxy
        
        def cached_dataset(idx):

            assert 0 <= idx <= 24, f"Penn Treebank only has section 00 to 24, while {str(idx)} found."
            
            idx = str(idx).zfill(2)
            
            if not (data_folder / "treebank_3").exists():
                # download and extract data
                penn_path = cached_path(web_path, Path("datasets") / dataset_name)
                unpack_file(penn_path, data_folder, "tar", True)
            
            # create dataset
            return PennTreebankDataset(
                path_to_mrg_folder=data_folder / "treebank_3" / "parsed" / "mrg" / "wsj" / idx,
                fileids='wsj_\d{4}.mrg'
            )
        
        # parse the section split numbers
        def getNum(s):
            l = []
            for r in s.strip().split(','):
                if '-' in r:
                    start, end = r.split('-')
                    l += [i for i in range(int(start), int(end)+1)]
                elif r.strip():
                    l.append(int(r))
            return l
        
        train, dev, test = map(getNum, splits.split('|'))
        print("Use Penn Treebank dataset with sections:")
        print("    train: {}, dev: {}, test: {}".format(*(splits.split('|'))))

        train_parts = [cached_dataset(i) for i in train]
        dev_parts = [cached_dataset(i) for i in dev]
        test_parts = [cached_dataset(i) for i in test]

        super(PennTreebankCorpus, self).__init__(
            ConcatDataset(train_parts) if len(train_parts) > 0 else None,
            ConcatDataset(dev_parts) if len(dev_parts) > 0 else None,
            ConcatDataset(test_parts) if len(test_parts) > 0 else None, 
            name=str(data_folder),
            sample_missing_splits=False
        )

    def make_tag_dictionary(self, tag_type: str) -> Dictionary:

        if tag_type == 'mlm':

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
        
        else:

            return super().make_tag_dictionary(tag_type)

class PennTreebankDataset(FlairDataset):
    def __init__(self, path_to_mrg_folder: Union[str, Path], fileids: Union[List, str] = ".*", in_memory: bool = True):
        """
        Instantiates a Penn Treebank dataset.

        :param path_to_mrg_folder: Path to the mrg file folder
        :param fileids: A list or regexp specifying the fileids in this corpus.
        :param in_memory: If set to True, keeps full dataset in memory, otherwise does disk reads
        """
        if type(path_to_mrg_folder) is str:
            path_to_mrg_folder = Path(path_to_mrg_folder)
        assert path_to_mrg_folder.exists()

        self.in_memory: bool = in_memory

        self.path_to_mrg_folder = path_to_mrg_folder
        self.total_sentence_count: int = 0

        # option 1: read only sentence boundaries as offset positions
        if not self.in_memory:
            raise NotImplementedError("Penn Treebank with disk reads is not implemented yet.")

        # option 2: keep everything in memory
        if self.in_memory:
            self.sentences: List[Sentence] = []

            READER = BracketParseCorpusReader(os.path.abspath(path_to_mrg_folder), fileids)
            for sent in READER.tagged_sents():
                sentence: Sentence = Sentence()
                for word, tag in sent:
                    if tag != '-NONE-':
                        token = Token(word)
                        token.add_label("pos", tag)
                        sentence.add_token(token)
                self.sentences.append(sentence)

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
            raise NotImplementedError("Penn Treebank with disk reads is not implemented yet.")

        return sentence

class BLLIPCorpus(Corpus):

    def __init__(self, 
                 base_path: Union[str, Path] = None, 
                 mode: str = "XS",
                 train_set_size: int = None,
                 dev_set_size: int = None,
                 test_set_size: int = None,
                 use_cache: bool = True,
                 in_memory: bool = False):
        """
        Instantiates a Corpus from BLLIP 1987-89 WSJ Corpus Release 1 dataset. This is only possible
        if you've manually downloaded the BLLIP datasets to your machine.
        Obtain the dataset from LDC (https://catalog.ldc.upenn.edu/LDC2000T43) and set base_path as 
        the path to the tar file.

        :param base_path: Base path for the dataset.
        :param mode: Pre-defined dataset splits. Options:
            - XS: train 40k, dev 20k, test 20k
            - SM: train 200k, dev 20k, test 20k
            - MD: train 600k, dev 20k, test 20k
            - LG: train 1756k, dev 20k, test 20k
            - FULL: train 81%, dev 9%, test 10%
            - CUSTOM: defined with parameters.
        :param train_set_size: Number of sentences in the randomly split training set.
        :param dev_set_size: Number of sentences in the randomly split dev set.
        :param test_set_size: Number of sentences in the randomly split test set.
        :param use_cache: Cache the dataset so that the second read will be much faster.
        :param in_memory: If set to True, keeps full dataset in memory, otherwise does disk reads
        :return: a Corpus with annotated train, dev and test data
        """

        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
            data_folder = base_path / dataset_name
        else:
            data_folder = base_path

        cache_path = Path(flair.cache_root) / "datasets" / dataset_name
        if not use_cache or not cache_path.exists():

            # check if data there
            if not data_folder.exists():
                log.warning("-" * 100)
                log.warning(f'WARNING: BLLIP corpus not found at "{data_folder}".')
                log.warning(
                    'Necessary data can be found here: "https://catalog.ldc.upenn.edu/LDC2000T43"'
                )
                log.warning("-" * 100)

            log.info("Caching BLLIPS dataset...")
            if not data_folder.is_dir():
                from tempfile import TemporaryDirectory
                with TemporaryDirectory() as tmp_path:
                    unpack_file(data_folder, tmp_path)
                    self.cache_dataset(Path(tmp_path) / "bliip_87_89_wsj")
            else:
                self.cache_dataset(data_folder)

        log.info("Loading BLLIPS dataset...")
        column_name_map = {1: "text", 2: "pos"}

        # dataset mode
        if mode == 'XS':
            train = ColumnDataset(cache_path / "train_xs.conll", column_name_map=column_name_map, column_delimiter="\t", in_memory=in_memory)
            dev = ColumnDataset(cache_path / "dev.conll", column_name_map=column_name_map, column_delimiter="\t", in_memory=in_memory)
            test = ColumnDataset(cache_path / "test.conll", column_name_map=column_name_map, column_delimiter="\t", in_memory=in_memory)
        elif mode == 'SM':
            train = ColumnDataset(cache_path / "train_sm.conll", column_name_map=column_name_map, column_delimiter="\t", in_memory=in_memory)
            dev = ColumnDataset(cache_path / "dev.conll", column_name_map=column_name_map, column_delimiter="\t", in_memory=in_memory)
            test = ColumnDataset(cache_path / "test.conll", column_name_map=column_name_map, column_delimiter="\t", in_memory=in_memory)
        elif mode == 'MD':
            train = ColumnDataset(cache_path / "train_md.conll", column_name_map=column_name_map, column_delimiter="\t", in_memory=in_memory)
            dev = ColumnDataset(cache_path / "dev.conll", column_name_map=column_name_map, column_delimiter="\t", in_memory=in_memory)
            test = ColumnDataset(cache_path / "test.conll", column_name_map=column_name_map, column_delimiter="\t", in_memory=in_memory)
        elif mode == 'LG':
            train = ColumnDataset(cache_path / "train_lg.conll", column_name_map=column_name_map, column_delimiter="\t", in_memory=in_memory)
            dev = ColumnDataset(cache_path / "dev.conll", column_name_map=column_name_map, column_delimiter="\t", in_memory=in_memory)
            test = ColumnDataset(cache_path / "test.conll", column_name_map=column_name_map, column_delimiter="\t", in_memory=in_memory)
        elif mode == 'FULL':
            train = ColumnDataset(cache_path / "raw.conll", column_name_map=column_name_map, column_delimiter="\t", in_memory=in_memory)
            dev = None
            test = None
        elif mode == 'CUSTOM':
            if test_set_size is None: test_set_size = 0
            if dev_set_size is None: dev_set_size = 0
            if train_set_size is None: train_set_size = 0

            dataset = ColumnDataset(cache_path / "raw.conll", column_name_map=column_name_map, column_delimiter="\t", in_memory=in_memory)

            # randomly split the dataset
            # We split test set first. If we set the random seed, we can obtain the same test
            # set whild changing the training set size.
            test, dataset = randomly_split_into_two_datasets(dataset, test_set_size)
            dev, dataset = randomly_split_into_two_datasets(dataset, dev_set_size)
            train, dataset = randomly_split_into_two_datasets(dataset, train_set_size)
        else:
            raise ValueError(f"Invalid mode: {mode}")

        super(BLLIPCorpus, self).__init__(
            train if len(train) > 0 else dataset,
            dev if (dev and len(dev) > 0) else None,
            test if (test and len(test) > 0) else None, 
            name=str(data_folder)
        )

    @classmethod
    def cache_dataset(cls, bllip_folder):

        from tqdm import tqdm
        from tempfile import TemporaryDirectory

        # this dataset name
        dataset_name = cls.__name__.lower()
        cache_path = Path(flair.cache_root) / "datasets" / dataset_name

        ## Pre-defined modes
        ## Follow the setting in https://github.com/yikangshen/UDGN/blob/1bd52ecc7b348ca591957f8001867a273e080929/data_dep.py
        path_1987 = '1987/w7_%03d'
        path_1988 = '1988/w8_%03d'
        path_1989 = '1989/w9_%03d'

        train_path_XS = [path_1987 % id for id in [71, 122]] + \
                        [path_1988 % id for id in [54, 107]] + \
                        [path_1989 % id for id in [28, 37]]

        train_path_SM = [path_1987 % id for id in [35, 43, 48, 54, 61, 71, 77, 81, 96, 122]] + \
                        [path_1988 % id for id in [24, 54, 55, 59, 69, 73, 76, 79, 90, 107]] + \
                        [path_1989 % id for id in [12, 13, 15, 18, 21, 22, 28, 37, 38, 39]]

        train_path_MD = [path_1987 % id for id in [5, 10, 18, 21, 22, 26, 32, 35, 43, 47, 48, 49, 51, 54, 55, 56, 57, 61, 62, 65, 71, 77, 79, 81, 90, 96, 100, 105, 122, 125]] + \
                        [path_1988 % id for id in [12, 13, 14, 17, 23, 24, 33, 39, 40, 47, 48, 54, 55, 59, 69, 72, 73, 76, 78, 79, 83, 84, 88, 89, 90, 93, 94, 96, 102, 107]] + \
                        [path_1989 % id for id in range(12, 42)]

        train_path_LG = [path_1987 % id for id in range(3, 128)] + \
                        [path_1988 % id for id in range(3, 109)] + \
                        [path_1989 % id for id in range(12, 42)]
        
        dev_path = ['1987/w7_001', '1988/w8_001', '1989/w9_010']
        test_path = ['1987/w7_002', '1988/w8_002', '1989/w9_011']

        def in_paths(this: Path, paths: List[str]):
            for path in paths:
                if this.samefile(bllip_folder / path):
                    return True
            return False

        # cache the dataset
        with TemporaryDirectory() as tmp_folder:

            tmp_folder = Path(tmp_folder)

            f_raw = open(tmp_folder / "raw.conll", "w+")
            f_train_xs = open(tmp_folder / "train_xs.conll", "w+")
            f_train_sm = open(tmp_folder / "train_sm.conll", "w+")
            f_train_md = open(tmp_folder / "train_md.conll", "w+")
            f_train_lg = open(tmp_folder / "train_lg.conll", "w+")
            f_dev = open(tmp_folder / "dev.conll", "w+")
            f_test = open(tmp_folder / "test.conll", "w+")

            def write_to(dataset, file):
                for sentence in dataset:
                    for i, token in enumerate(sentence):
                        print(i+1, token.text, token.get_tag('pos').value, sep='\t', file=file)
                    print(file=file)
        
            for folder_name in tqdm(['1987', '1988', '1989'], leave=False, desc="Year"):

                folder: Path = bllip_folder / folder_name
                subfolders: List[Path] = sorted(list(folder.iterdir()), key=lambda x: str(x))

                for subfolder in tqdm(subfolders, total=len(subfolders), leave=False, desc="Section"):

                    assert subfolder.is_dir(), f"'{str(subfolder)}' is not a directory."

                    # create dataset
                    dataset = PennTreebankDataset(path_to_mrg_folder=subfolder)
                    
                    if in_paths(subfolder, train_path_XS):
                        write_to(dataset, f_train_xs)
                    if in_paths(subfolder, train_path_SM):
                        write_to(dataset, f_train_sm)
                    if in_paths(subfolder, train_path_MD):
                        write_to(dataset, f_train_md)
                    if in_paths(subfolder, train_path_LG):
                        write_to(dataset, f_train_lg)
                    if in_paths(subfolder, dev_path):
                        write_to(dataset, f_dev)
                    if in_paths(subfolder, test_path):
                        write_to(dataset, f_test)
                    write_to(dataset, f_raw)

            def fsync(file, filename):
                with open(cache_path / filename, "w") as wf:
                    file.seek(0)
                    wf.write(file.read())

            os.makedirs(cache_path, exist_ok=True)
            fsync(f_raw, "raw.conll")
            fsync(f_train_xs, "train_xs.conll")
            fsync(f_train_sm, "train_sm.conll")
            fsync(f_train_md, "train_md.conll")
            fsync(f_train_lg, "train_lg.conll")
            fsync(f_dev, "dev.conll")
            fsync(f_test, "test.conll")
        