import logging
from pathlib import Path
from typing import List, Union
import os

from torch.utils.data.dataset import ConcatDataset, Subset

import flair
from flair.data import (
    Sentence,
    Corpus,
    Token,
    FlairDataset,
    randomly_split_into_two_datasets
)
from flair.datasets import UniversalDependenciesCorpus
from flair.file_utils import cached_path, unpack_file

import nltk
from nltk.corpus.reader.bracket_parse import BracketParseCorpusReader

log = logging.getLogger("flair")

class PennTreebankCorpus(Corpus):
    def __init__(self, splits: str = "2-21|22|23", base_path: Union[str, Path] = None, in_memory: bool = True):
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
                 train_set_size: int = 600000,
                 dev_set_size: int = 20000,
                 test_set_size: int = 20000,
                 in_memory: bool = True):
        """
        Instantiates a Corpus from BLLIP 1987-89 WSJ Corpus Release 1 dataset. This is only possible
        if you've manually downloaded the BLLIP datasets to your machine.
        Obtain the dataset from LDC (https://catalog.ldc.upenn.edu/LDC2000T43) and extract the tar 
        file. Then set the base_path parameter in the constructor to the path to the extracted directory
        called 'bliip_87_89_wsj'. It should have at least 3 folders in it: '1987', '1988' and '1989'.

        :param base_path: Base path for flair storage. Usually use default setting.
        :param train_set_size: Number of sentences in the randomly split training set.
        :param dev_set_size: Number of sentences in the randomly split dev set.
        :param test_set_size: Number of sentences in the randomly split test set.
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

        # check if data there
        if not data_folder.exists():
            log.warning("-" * 100)
            log.warning(f'WARNING: BLLIP corpus not found at "{data_folder}".')
            log.warning(
                'Necessary data can be found here: "https://catalog.ldc.upenn.edu/LDC2000T43"'
            )
            log.warning("-" * 100)
        
        # read the dataset
        datasets = []
        
        for folder_name in ['1987', '1988', '1989']:

            folder = data_folder / folder_name

            for subfolder in folder.iterdir():

                assert subfolder.is_dir(), f"'{str(subfolder)}' is not a directory."

                # create dataset
                dataset = PennTreebankDataset(path_to_mrg_folder=subfolder)
                datasets.append(dataset)
        
        dataset = ConcatDataset(datasets)
        log.info("Number of sentences in the BLLIP Corpus:", len(dataset))

        # randomly split the dataset
        # We split test set first. If we set the random seed, we can obtain the same test
        # set whild changing the training set size.
        test_parts, dataset = randomly_split_into_two_datasets(dataset, test_set_size)
        dev_parts, dataset = randomly_split_into_two_datasets(dataset, dev_set_size)
        train_parts, dataset = randomly_split_into_two_datasets(dataset, train_set_size)

        super(BLLIPCorpus, self).__init__(
            train_parts if len(train_parts) > 0 else None,
            dev_parts if len(dev_parts) > 0 else None,
            test_parts if len(test_parts) > 0 else None, 
            name=str(data_folder),
            sample_missing_splits=False
        )