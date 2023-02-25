from .treebanks import PennTreebankCorpus
from .treebanks import PennTreebankDataset
from .treebanks import BLLIPCorpus
from .treebanks import CFQ_Dependency
from .sequence_labeling import AUTO_CONLL_03, COGS_SequenceLabeling

from .text import StandardPTBCorpus, BLLIPTextCorpus, UDGNBLLIPTextCorpus

from .text_classification import SENTEVAL_SST_BINARY, SENTEVAL_SST_GRANULAR

from .toy import ToyCorpus