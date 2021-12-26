from os import rename
from pathlib import Path
import flair
from flair.datasets import CONLL_03
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