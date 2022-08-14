## ================= MODIFY THE SETTING HERE =================
SENTENCE = "The quick brown fox jumped over the lazy dog ."
ALGORITHM = 'projective' # 'argmax', 'nonprojective', 'projective'
MODE = 'all' # 'all', 'average'
## ===========================================================

from pathlib import Path
from typing import List
import os

if 'DRAW' not in os.environ:
    os.environ.setdefault('DRAW', '1')


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-c","--config", help="Determine the config file.", dest="config", type=str, default="config.toml")
parser.add_argument("-f","--force", help="Force training to start, bypass the existence check.", dest="force", action='store_true')
args = parser.parse_args()


import utils
config = utils.readconfig(args.config)
base_path = Path(config["Basic"]["base_path"]) / config["Basic"]["folder_name"]
utils.set_cuda_id(config["Basic"]["cuda_id"])


import torch
torch.set_num_threads(4)

import flair
flair.set_seed(config["Basic"]["random_seed"])
# flair.device = torch.device("cpu")

import flair.datasets
import flair.embeddings
from flair.data import Sentence

import models

# initialize sequence tagger
model_name = config["SequenceTagger"].pop("tagger", "CustomSequenceTagger")
modelClass = utils.getattrs([models], model_name)
tagger = modelClass.load(base_path / "best-model.pt")

sentence = Sentence(SENTENCE, use_tokenizer=False)
tagger.predict(sentence)

# generate .tex file
with open(base_path / "dep_graph.tex", 'w') as f:
    recorder = tagger.mod.recorder
    print(recorder.export_latex_codes(sentence, algorithm = ALGORITHM, mode = MODE), file=f)
