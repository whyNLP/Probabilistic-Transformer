from pathlib import Path
from typing import List, Dict, Union, Callable

import flair
from flair.data import (
    Sentence,
    Corpus,
    Token,
    Dictionary,
    Tokenizer
)
from flair.tokenization import SegtokTokenizer, SpaceTokenizer
from flair.datasets import ClassificationCorpus
from flair.file_utils import cached_path, unzip_file

from flair.datasets import SENTEVAL_SST_GRANULAR as SENTEVAL_SST_GRANULAR_

class SENTEVAL_SST_BINARY(ClassificationCorpus):
    """
    The Stanford sentiment treebank dataset of SentEval, see https://github.com/facebookresearch/SentEval, classified
    into NEGATIVE or POSITIVE sentiment.
    """

    def __init__(
            self,
            tokenizer: Union[bool, Callable[[str], List[Token]], Tokenizer] = SpaceTokenizer(),
            memory_mode: str = 'full',
            **corpusargs,
    ):
        """
        Instantiates SentEval Stanford sentiment treebank dataset.
        :param memory_mode: Set to 'full' by default since this is a small corpus. Can also be 'partial' or 'none'.
        :param tokenizer: Custom tokenizer to use (default is SpaceTokenizer)
        :param corpusargs: Other args for ClassificationCorpus.
        """

        # this dataset name
        dataset_name = self.__class__.__name__.lower() + '_v2'

        # default dataset folder is the cache root
        data_folder = Path(flair.cache_root) / "datasets" / dataset_name

        # download data if necessary
        if not (data_folder / "train.txt").is_file():

            # download senteval datasets if necessary und unzip
            cached_path('https://raw.githubusercontent.com/PrincetonML/SIF/master/data/sentiment-train',
                        Path("datasets") / dataset_name / 'raw')
            cached_path('https://raw.githubusercontent.com/PrincetonML/SIF/master/data/sentiment-test',
                        Path("datasets") / dataset_name / 'raw')
            cached_path('https://raw.githubusercontent.com/PrincetonML/SIF/master/data/sentiment-dev',
                        Path("datasets") / dataset_name / 'raw')

            # create train.txt file by iterating over pos and neg file
            with open(data_folder / "train.txt", "a") as out_file, open(
                    data_folder / 'raw' / "sentiment-train") as in_file:
                for line in in_file:
                    fields = line.split('\t')
                    label = 'POSITIVE' if fields[1].rstrip() == '1' else 'NEGATIVE'
                    out_file.write(f"__label__{label} {fields[0]}\n")

            # create dev.txt file by iterating over pos and neg file
            with open(data_folder / "dev.txt", "a") as out_file, open(
                    data_folder / 'raw' / "sentiment-dev") as in_file:
                for line in in_file:
                    fields = line.split('\t')
                    label = 'POSITIVE' if fields[1].rstrip() == '1' else 'NEGATIVE'
                    out_file.write(f"__label__{label} {fields[0]}\n")

            # create test.txt file by iterating over pos and neg file
            with open(data_folder / "test.txt", "a") as out_file, open(
                    data_folder / 'raw' / "sentiment-test") as in_file:
                for line in in_file:
                    fields = line.split('\t')
                    label = 'POSITIVE' if fields[1].rstrip() == '1' else 'NEGATIVE'
                    out_file.write(f"__label__{label} {fields[0]}\n")

        super(SENTEVAL_SST_BINARY, self).__init__(
            data_folder,
            tokenizer=tokenizer,
            memory_mode=memory_mode,
            **corpusargs,
        )

    def make_tag_dictionary(self, tag_type: str = None) -> Dictionary:
        return super(SENTEVAL_SST_BINARY, self).make_label_dictionary(tag_type)

class SENTEVAL_SST_GRANULAR(SENTEVAL_SST_GRANULAR_):
    def make_tag_dictionary(self, tag_type: str = None) -> Dictionary:
        return super(SENTEVAL_SST_GRANULAR, self).make_label_dictionary(tag_type)