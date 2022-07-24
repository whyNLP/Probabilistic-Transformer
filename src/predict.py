from collections import Counter
import logging
from pathlib import Path
from typing import List


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-c","--config", help="Determine the config file.", dest="config", type=str, default="config.toml")
parser.add_argument("-f","--force", help="Force training to start, bypass the existence check.", dest="force", action='store_true')
args = parser.parse_args()


import utils
config = utils.readconfig(args.config)
base_path = Path(config["Basic"]["base_path"]) / config["Basic"]["folder_name"]
if (not args.force) and ((base_path / "best-model.pt").exists() or (base_path / "config.toml").exists()):
    if input("WARNING: Overwrite existing model. Continue? y/[n]: ") != 'y':
        exit(-1)
utils.set_cuda_id(config["Basic"]["cuda_id"])


import torch
torch.set_num_threads(4)

import flair
flair.set_seed(config["Basic"]["random_seed"])
# flair.device = torch.device("cpu")

import flair.datasets
from flair.data import Corpus, Dictionary, Sentence
import flair.embeddings
from flair.training_utils import add_file_handler

import models
log = logging.getLogger("flair")

# 5. initialize sequence tagger
model_name = config["SequenceTagger"].pop("tagger", "CustomSequenceTagger")
modelClass = utils.getattrs([models], model_name)
tagger = modelClass.load(base_path / "best-model.pt")

sentence = Sentence(["no", "it", "was", "n't", "black", "monday"])
tagger.predict(sentence)

# Uncomment if you need label
# tagger.predict(sentence, obtain_labels=True, all_tag_prob=True)

print(sentence)
