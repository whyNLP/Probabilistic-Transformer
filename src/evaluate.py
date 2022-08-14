"""
Evaluate on PTB CONLL dataset for unsupervised dependency parsing.

MODE:
 - 'average': Use the average of all channels' probabilities as the score matrix.
 - 'hit': Each channel produce one parse tree. So each word has multiple head options. If any one hits the gold answer, then we take it as correct.
 - 'best': Each channel produce one parse tree. We evaluate them seperately, then choose the best channel's result as final result.
 - 'left': All left arcs.
 - 'right': All right arcs.
 - 'random': Random heads.
"""
## ================= MODIFY THE SETTING HERE =================
MODE = 'average' # Options: 'average', 'hit', 'best', 'left', 'right', 'random'
ITERATION = -1 # Options: 1, 2, ...; -1 is the last iteration.
ALGORITHM = 'argmax' # 'argmax', 'nonprojective', 'projective'
## ===========================================================

from pathlib import Path
from typing import List
from tqdm import tqdm
import random
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

from flair.datasets import ColumnDataset
import flair.embeddings
from flair.data import Sentence
from flair.training_utils import Metric

import models

# load dataset
test_dataset = ColumnDataset("/public/home/wuhy1/projects/tmp/wsj_sd33/test.gold.conllu", column_name_map={1: 'text', 6: 'head'})

# initialize sequence tagger
model_name = config["SequenceTagger"].pop("tagger", "CustomSequenceTagger")
modelClass = utils.getattrs([models], model_name)
tagger = modelClass.load(base_path / "best-model.pt")


metric = Metric("Evaluation")
if MODE == 'average':
    for sentence in tqdm(test_dataset, leave=False):
        tagger.predict(sentence)
        recorder = tagger.mod.recorder

        heads = recorder[ITERATION]
        _, pred = recorder.get_dependency_heads(heads, ALGORITHM)
        gold = [int(token.get_tag('head').value) for token in sentence]
        for p, g in zip(pred, gold):
            if p == g:
                metric.add_tp('UAS')
            else:
                metric.add_fn('UAS')
elif MODE == 'hit':
    for sentence in tqdm(test_dataset, leave=False):
        tagger.predict(sentence)
        recorder = tagger.mod.recorder
        gold = [int(token.get_tag('head').value) for token in sentence]

        tp = 0
        for head in recorder.heads:
            heads = recorder.retrieve_probs(ITERATION, head=head)
            _, pred = recorder.get_dependency_heads(heads, ALGORITHM)
            tp = max(tp, [p == g for p, g in zip(pred, gold)].count(True))
        fn = len(sentence) - tp
        metric._tps['UAS'] += tp
        metric._fns['UAS'] += fn
elif MODE == 'best':
    # dummy sentence to get heads
    tagger.predict(Sentence("I love NLP"))
    recorder = tagger.mod.recorder
    heads = recorder.heads

    metrics = [Metric("Evaluation") for _ in range(len(heads))]
    for sentence in tqdm(test_dataset, leave=False):
        tagger.predict(sentence)
        recorder = tagger.mod.recorder
        gold = [int(token.get_tag('head').value) for token in sentence]

        for head, metric in zip(heads, metrics):
            probs = recorder.retrieve_probs(ITERATION, head=head)
            _, pred = recorder.get_dependency_heads(probs, ALGORITHM)
            for p, g in zip(pred, gold):
                if p == g:
                    metric.add_tp('UAS')
                else:
                    metric.add_fn('UAS')
        
    accuracys = [m.accuracy() for m in metrics]
    metric = [m for m in metrics if m.accuracy() == max(accuracys)][0]
elif MODE == 'right':
    for sentence in tqdm(test_dataset, leave=False):
        pred = list(range(len(sentence)))
        gold = [int(token.get_tag('head').value) for token in sentence]
        for p, g in zip(pred, gold):
            if p == g:
                metric.add_tp('UAS')
            else:
                metric.add_fn('UAS')
elif MODE == 'left':
    for sentence in tqdm(test_dataset, leave=False):
        pred = list(range(2, len(sentence)+1)) + [0]
        gold = [int(token.get_tag('head').value) for token in sentence]
        for p, g in zip(pred, gold):
            if p == g:
                metric.add_tp('UAS')
            else:
                metric.add_fn('UAS')
elif MODE == 'random':
    for sentence in tqdm(test_dataset, leave=False):
        pred = [random.randint(0, len(sentence)) for token in sentence]
        gold = [int(token.get_tag('head').value) for token in sentence]
        for p, g in zip(pred, gold):
            if p == g:
                metric.add_tp('UAS')
            else:
                metric.add_fn('UAS')
else:
    raise NotImplementedError

print()
print("Dependency UAS:", metric.accuracy())