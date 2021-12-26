from collections import Counter
import logging
from pathlib import Path
from typing import List


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-c","--config", help="Determine the config file.", dest="config", type=str, default="config.toml")
args = parser.parse_args()


import utils
config = utils.readconfig(args.config)
base_path = Path(config["Basic"]["base_path"]) / config["Basic"]["folder_name"]
if (base_path / "best-model.pt").exists() or (base_path / "config.toml").exists():
    if input("WARNING: Overwrite existing model. Continue? y/[n]: ") != 'y':
        exit(-1)
utils.saveconfig(args.config, base_path / "config.toml")
utils.set_cuda_id(config["Basic"]["cuda_id"])


import torch
torch.set_num_threads(4)

import flair
flair.set_seed(config["Basic"]["random_seed"])
# flair.device = torch.device("cpu")

import flair.datasets
from flair.data import Corpus, Dictionary
from flair.embeddings import (
    TokenEmbeddings,
    StackedEmbeddings,
    OneHotEmbeddings
)
import flair.embeddings
from flair.training_utils import EvaluationMetric, add_file_handler

import embeddings
import datasets
import models
import trainers

log = logging.getLogger("flair")
log_handler = add_file_handler(log, base_path / "console.log")


# 1. what tag do we want to predict?
tag_type = config["Corpus"].pop("tag_type")


# 2. get the corpus
corpus_name, corpus_config = list(config["Corpus"].items())[0]
corpus: Corpus = utils.getattrs([datasets, flair.datasets], corpus_name)(**corpus_config)
log.info(corpus)


# 3. make the tag dictionary from the corpus
if tag_type == 'mlm':
    ## Specially take care of MLM
    min_freq = config["Embeddings"]["MLMOneHotEmbeddings"]["min_freq"]

    tag_dictionary = Dictionary()
    tokens = list(map((lambda s: s.tokens), corpus.train))
    tokens = [token for sublist in tokens for token in sublist]
    most_common = Counter(list(map((lambda t: t.text), tokens)))
    tokens = [token for token, freq in most_common.items() if freq >= min_freq]
    for token in tokens: tag_dictionary.add_item(token)
else:
    tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
log.info(tag_dictionary)


# 4. select embeddings
embedding_types: List[TokenEmbeddings] = []
if "OneHotEmbeddings" in config["Embeddings"]:
    import sys, os
    sys.stdout = open(os.devnull, 'w')
    word_embedding = OneHotEmbeddings(**(config["Embeddings"].pop("OneHotEmbeddings")), corpus=corpus)
    word_embedding.instance_parameters["corpus"] = corpus_name # FIXME: this is a quick fix for error in saving conll03.
    embedding_types.append(word_embedding)
    sys.stdout = sys.__stdout__
    log.info(f"OneHotEmbeddings: Vocab size = {len(word_embedding.vocab_dictionary)}")
if "MLMOneHotEmbeddings" in config["Embeddings"]:
    word_embedding = embeddings.MLMOneHotEmbeddings(**(config["Embeddings"].pop("MLMOneHotEmbeddings")), corpus=corpus)
    embedding_types.append(word_embedding)
for k, v in config["Embeddings"].items():
    word_embedding = utils.getattrs([embeddings, flair.embeddings], k)(**v)
    embedding_types.append(word_embedding)

embeddings_: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)
log.removeHandler(log_handler)


# 5. initialize sequence tagger
model_name = config["SequenceTagger"].pop("tagger", "CustomSequenceTagger")
modelClass = utils.getattrs([models], model_name)
tagger = modelClass(
    embeddings=embeddings_,
    tag_dictionary=tag_dictionary,
    tag_type=tag_type,
    **config["SequenceTagger"]
).load(base_path / "best-model.pt")


# 6. initialize trainer
trainer_name = config["Trainer"].pop("trainer", "CustomModelTrainer")
trainerClass = utils.getattrs([trainers], trainer_name)
trainer = trainerClass(tagger, corpus, optimizer=torch.optim.Adam, **(config["Trainer"].pop("init") if "init" in config["Trainer"] else {}) )

if "scheduler" in config["Trainer"]:
    config["Trainer"]["scheduler"] = eval(config["Trainer"]["scheduler"])


# 7. train
trainer.train(
    base_path,
    **config["Trainer"],
)
