# Probabilistic Transformer

The code base for project **Probabilistic Transformer**, a model of contextual word representation from a syntactic and probabilistic perspective.

> **Warning**  
> This git branch is only for reproducing the results. The codes are developed in a way that is easy to integrate with all kinds of modules, but not well-optimized for speed. The repo structure is a bit messy. The framework it uses ([flair](https://github.com/flairNLP/flair)) is outdated. Further development based on this branch is not encouraged.

## Preparation

### Code Environment

To prepare the code environment, use

```sh
cd src
pip install -r requirements.txt
```

Due to package compatibility, it will install `pytorch` with version `1.7.1`. Feel free to upgrade it with the command:

```sh
pip install torch==1.10.2
```

Or this command:

```sh
pip install --upgrade torch
```

This work is developed under `torch==1.10.2`.

### Dataset

Our code will automatically download the dataset if it finds the dataset you want to use is missing. Some datasets require license/purchase, and the code would throw an error telling you where to download the dataset. We also provide detailed instructions in the [template config file](src/config.toml.template) and doc strings.

## How to run

### Training

Simply run the following commands:

```sh
cd src
python train.py
```

By default, it will use the config file `src/config.toml`. To use other config files, use `-c` or `--config` to specify the configuration file:

```sh
cd src
python train.py -c ../path/to/config.toml
```

### Prediction / Inference

To do inference on a sentence, run `python predict.py`. The usage is exactly the same with training, just use a config that has just been used for training before. To modify the sentence for inference, please modify codes in `predict.py`.

### Drawing Dependency Parse Trees

To visualize the dependency parse trees produced by our models, run `python draw.py`. The usage is the same as inference. It will generate `dep_graph.tex` in your working directory. You may compile the latex file and get the figures in PDF.

There are 3 options at the top of the file `draw.py`:
- `SENTENCE`: The sentence for dependency parsing. It will be tokenized with a white space tokenizer.
- `ALGORITHM`: The algorithm for generating dependency parse trees. Options: `argmax`, `nonprojective`, `projective`.
  - `argmax`: Use the most probable head for each token. It doesn't care whether the generated parse tree is connected or not.
  - `nonprojective`: Parse with the Chu-Liu-Edmonds algorithm. The produced parse tree is valid but not necessarily projective. If `ROOT` is not considered in the model (`[CLS]` could be seen as a counterpart of `ROOT`), then all scores for `ROOT` will be zero.
  - `projective`: Parse with the Eisner algorithm. The produced parse tree is projective. If `ROOT` is not considered in the model (`[CLS]` could be seen as a counterpart of `ROOT`), then all scores for `ROOT` will be zero.
- `MODE`: Whether to combine results in different heads (transformers) / channels (probabilistic transformers). Options: `all`, `average`:
  - `all`: Draw parse trees for each layer/iteration and each head/channel.
  - `average`: Draw parse trees for each layer/iteration and use the average score in different heads/channels as the score in this layer/iteration.

If we take the attention scores in transformers as the dependency edge scores, then we may also draw dependency parse trees from transformers.

### Evaluation for Unsupervised Dependency Parsing Task

To do unsupervised dependency parsing, run `python evaluate.py`. The usage is the same as drawing. It will print the UAS (Unlabeled Attachment Score) to the console.

There are 4 options at the top of the file `draw.py`:
- `TEST_FILE`: Path to conll-style file as the dependency parsing test set. None for UD-English ewt test set. Type: None or str.
- `ALGORITHM`: Same as drawing.
- `MODE`: How to do evaluation.
  - `average`: Use the average of all channels' probabilities as the score matrix.
  - `hit`: Each channel produce one parse tree. So each word has multiple head options. If any one hits the gold answer, then we take it as correct.
  - `best`: Each channel produce one parse tree. We evaluate them seperately, then choose the best channel's result as final result.
  - `left`: All left arcs.
  - `right`: All right arcs.
  - `random`: Random heads.
- `ITERATION`: Use the dependency head distribution from which iteration. Use numbers 1, 2, ... or -1 for the last iteration.

## Result

We provide the config files in `configs/best`. To reproduce the results, please use the following command

```sh
cd src
python train.py -c ../configs/best/<CONFIG_FILE>
```

where `<CONFIG_FILE>` should be replaced by the config file in the tables below.

> **Note**  
> Part of the results presented below was not contained in our paper.

### Probabilistic Transformer

| Task  | Dataset  |   Metric   |         Config          |      Performance (avg. 5 runs)      | # of Parameters | Speed (Sample/sec) | Total Time |
| :---: | :------: | :--------: | :---------------------: | :---------------------------------: | :-------------: | :----------------: | :--------: |
|  MLM  |   PTB    | Perplexity |   `crf-mlm-ptb.toml`    |          62.86 $\pm$ 0.40           |     6291456     |       173.95       |  15:26:53  |
|  MLM  | BLLIP-XS | Perplexity |  `crf-mlm-bllip.toml`   |          123.18 $\pm$ 1.50          |     6291456     |       172.01       |  20:30:13  |
|  POS  |   PTB    |  Accuracy  |   `crf-pos-ptb.toml`    |          96.29 $\pm$ 0.03           |     3145728     |       222.91       |  5:13:42   |
|  POS  |    UD    |  Accuracy  |    `crf-pos-ud.toml`    |          90.96 $\pm$ 0.10           |     2359296     |       385.84       |  1:02:42   |
| UPOS  |    UD    |  Accuracy  |   `crf-upos-ud.toml`    |          91.57 $\pm$ 0.12           |     4194304     |       205.83       |  1:47:38   |
|  NER  | CONLL03  |     F1     | `crf-ner-conll03.toml`  |          75.47 $\pm$ 0.35           |     9437184     |       202.84       |  2:45:25   |
|  CLS  |  SST-2   |  Accuracy  |   `crf-cls-sst2.toml`   |          82.04 $\pm$ 0.88           |    10485760     |       675.78       |  1:54:03   |
|  CLS  |  SST-5   |  Accuracy  |   `crf-cls-sst5.toml`   |          42.77 $\pm$ 1.18           |     2630656     |       185.33       |  1:13:36   |
|  SYN  |   COGS   |  Accuracy  |   `crf-syn-cogs.toml`   |          84.60 $\pm$ 2.06           |     147456      |       507.66       |  2:14:25   |
|  SYN  | CFQ-mcd1 |  EM / LAS  | `crf-syn-cfq-mcd1.toml` | 78.88 $\pm$ 2.81 / 97.84 $\pm$ 0.33 |     1114112     |       234.13       |  19:04:35  |
|  SYN  | CFQ-mcd2 |  EM / LAS  | `crf-syn-cfq-mcd2.toml` | 48.41 $\pm$ 4.99 / 91.91 $\pm$ 0.68 |     1114112     |       225.75       |  19:22:46  |
|  SYN  | CFQ-mcd3 |  EM / LAS  | `crf-syn-cfq-mcd3.toml` | 45.68 $\pm$ 4.17 / 90.87 $\pm$ 0.70 |     1114112     |       269.96       |  14:26:53  |

### Transformer

| Task  | Dataset  |   Metric   |             Config              |      Performance (avg. 5 runs)      | # of Parameters | Speed (Sample/sec) | Total Time |
| :---: | :------: | :--------: | :-----------------------------: | :---------------------------------: | :-------------: | :----------------: | :--------: |
|  MLM  |   PTB    | Perplexity |   `transformer-mlm-ptb.toml`    |          58.43 $\pm$ 0.58           |    23809408     |       434.90       |  6:27:05   |
|  MLM  | BLLIP-XS | Perplexity |  `transformer-mlm-bllip.toml`   |          101.91 $\pm$ 1.40          |    11678720     |       616.84       |  7:10:23   |
|  POS  |   PTB    |  Accuracy  |   `transformer-pos-ptb.toml`    |          96.44 $\pm$ 0.04           |    15358464     |       527.46       |  2:11:05   |
|  POS  |    UD    |  Accuracy  |    `transformer-pos-ud.toml`    |          91.17 $\pm$ 0.11           |     3155456     |       554.10       |  0:39:34   |
| UPOS  |    UD    |  Accuracy  |   `transformer-upos-ud.toml`    |          91.96 $\pm$ 0.06           |    14368256     |       696.49       |  0:31:52   |
|  NER  | CONLL03  |     F1     | `transformer-ner-conll03.toml`  |          74.02 $\pm$ 1.11           |     1709312     |       577.57       |  0:49:38   |
|  CLS  |  SST-2   |  Accuracy  |   `transformer-cls-sst2.toml`   |          82.51 $\pm$ 0.26           |    23214080     |       713.34       |  2:03:30   |
|  CLS  |  SST-5   |  Accuracy  |   `transformer-cls-sst5.toml`   |          40.13 $\pm$ 1.09           |     8460800     |       871.61       |  0:17:42   |
|  SYN  |   COGS   |  Accuracy  |   `transformer-syn-cogs.toml`   |          82.05 $\pm$ 2.18           |     100000      |       856.28       |  1:16:25   |
|  SYN  | CFQ-mcd1 |  EM / LAS  | `transformer-syn-cfq-mcd1.toml` | 92.35 $\pm$ 2.37 / 99.21 $\pm$ 0.30 |     1189728     |       618.95       |  7:33:43   |
|  SYN  | CFQ-mcd2 |  EM / LAS  | `transformer-syn-cfq-mcd2.toml` | 80.34 $\pm$ 1.40 / 96.24 $\pm$ 0.68 |     1189728     |       590.35       |  8:15:08   |
|  SYN  | CFQ-mcd3 |  EM / LAS  | `transformer-syn-cfq-mcd3.toml` | 73.43 $\pm$ 6.07 / 94.85 $\pm$ 0.93 |     1189728     |       601.13       |  8:29:28   |

### Universal Transformer

| Task  | Dataset  |  Metric  |                  Config                   |      Performance (avg. 5 runs)      | # of Parameters | Speed (Sample/sec) | Total Time |
| :---: | :------: | :------: | :---------------------------------------: | :---------------------------------: | :-------------: | :----------------: | :--------: |
|  SYN  |   COGS   | Accuracy |   `universal-transformer-syn-cogs.toml`   |          80.50 $\pm$ 3.49           |      50000      |      1008.65       |  1:15:29   |
|  SYN  | CFQ-mcd1 | EM / LAS | `universal-transformer-syn-cfq-mcd1.toml` | 95.48 $\pm$ 2.09 / 99.59 $\pm$ 0.19 |     198288      |       603.01       |  8:20:50   |
|  SYN  | CFQ-mcd2 | EM / LAS | `universal-transformer-syn-cfq-mcd2.toml` | 78.63 $\pm$ 3.54 / 95.62 $\pm$ 0.75 |     198288      |       626.53       |  9:07:15   |
|  SYN  | CFQ-mcd3 | EM / LAS | `universal-transformer-syn-cfq-mcd3.toml` | 71.49 $\pm$ 5.39 / 94.57 $\pm$ 1.25 |     198288      |       603.23       |  8:17:17   |


<sub><i>* "Universal Transformer" only means weight sharing between layers in transformers. See details in [Ontanón et al. (2021)](https://aclanthology.org/2022.acl-long.251).</i></sub>  
<sub><i>** The training speed and time are for reference only. The speed data is randomly picked during the training and the product of speed and time is not equal to the number of samples.</i></sub>  
<sub><i>*** The random seeds for the 5 runs are: 0, 1, 2, 3, 4.</i></sub>

## Questions

> 1. I am working on a cluster where the compute node does not have Internet, so I cannot download the dataset before training. What should I do?

That is simple. Go to `src/train.py` and add `exit(0)` before training (line 105). Execute the training command in the login node (where you have access to the Internet). It will download the dataset without training the model. Finally, remove the line of code you added and train the model in the compute node.

> 2. Why not test on the GLUE dataset?

GLUE is a standard benchmark for language understanding, and most recent works with strong pre-trained word representations choose to test their models on this dataset. Our work does not involve pre-training, which indicates a weak ability for language understanding. To better evaluate the ability of word representation for our model, we think it might be more suitable to compare our model with a vanilla transformer on MLM and POS tagging tasks than GLUE.

> 3. Probabilistic Transformers take much more time to converge than transformers. Why?

Actually, experiments show that Probabilistic Transformers converge much faster than transformers in the early stage. For example, after 10 epochs of training on the MLM task PTB dataset, Probabilistic Transformers reach a perplexity of 194.69 on the PTB dev set, while transformers have a perplexity of 239.06. However, at the end of the training transformers could converge to a better local optimum. In addition, due to the inefficient positional representation, Probabilistic Transformers are slower to train than transformers (for the same number of training samples).

> 4. How strong is your baseline?

To make sure our baseline (transformer) implementation is strong enough, part of our experiments use the same setting as previous works:
- Our experiment for task MLM, dataset PTB has the same setting as that of [StructFormer](https://aclanthology.org/2021.acl-long.559/).
- Our experiment for task MLM, dataset BLLIP-XS has the same setting as that of [UDGN](https://aclanthology.org/2022.acl-long.327), though they did not conduct experiments on this split. The reason we do not follow StructFormer is that the code for this dataset is [not open-sourced](https://github.com/google-research/google-research/tree/master/structformer).
- Our experiment for task SYN, dataset COGS has the same setting as that of [Compositional](https://aclanthology.org/2022.acl-long.251).
- Our experiment for task SYN, dataset CFQ has the same setting as that of [Edge Transformer](https://proceedings.neurips.cc/paper/2021/hash/0a4dc6dae338c9cb08947c07581f77a2-Abstract.html).

<details>

<summary>Details for Baseline Compariason</summary>

| Task  | Dataset  |   Metric   | Source                                                                                                                                       |             Performance             |
| :---: | :------: | :--------: | :------------------------------------------------------------------------------------------------------------------------------------------- | :---------------------------------: |
|  MLM  |   PTB    | Perplexity | Transformer, [Shen et al. (2021)](https://aclanthology.org/2021.acl-long.559/)                                                               |                64.05                |
|  MLM  |   PTB    | Perplexity | Structformer, [Shen et al. (2021)](https://aclanthology.org/2021.acl-long.559/)                                                              |                60.94                |
|  MLM  |   PTB    | Perplexity | Transformer, Ours                                                                                                                            |                58.43                |
|  SYN  |   COGS   |  Accuracy  | Universal Transformer, [Ontanón et al. (2021)](https://aclanthology.org/2022.acl-long.251)                                                   |                78.4                 |
|  SYN  |   COGS   |  Accuracy  | Transformer, Ours                                                                                                                            |                82.05                |
|  SYN  |   COGS   |  Accuracy  | Universal Transformer, Ours                                                                                                                  |                80.50                |
|  SYN  | CFQ-mcd1 |  EM / LAS  | Transformer, [Bergen et al. (2021)](https://proceedings.neurips.cc/paper/2021/hash/0a4dc6dae338c9cb08947c07581f77a2-Abstract.html)           |   75.3 $\pm$ 1.7 / 97.0 $\pm$ 0.1   |
|  SYN  | CFQ-mcd1 |  EM / LAS  | Transformer, Ours                                                                                                                            | 92.35 $\pm$ 2.37 / 99.21 $\pm$ 0.30 |
|  SYN  | CFQ-mcd1 |  EM / LAS  | Universal Transformer, [Bergen et al. (2021)](https://proceedings.neurips.cc/paper/2021/hash/0a4dc6dae338c9cb08947c07581f77a2-Abstract.html) |   80.1 $\pm$ 1.7 / 97.8 $\pm$ 0.2   |
|  SYN  | CFQ-mcd1 |  EM / LAS  | Universal Transformer, Ours                                                                                                                  | 95.48 $\pm$ 2.09 / 99.59 $\pm$ 0.19 |
|  SYN  | CFQ-mcd2 |  EM / LAS  | Transformer, [Bergen et al. (2021)](https://proceedings.neurips.cc/paper/2021/hash/0a4dc6dae338c9cb08947c07581f77a2-Abstract.html)           |   59.3 $\pm$ 2.7 / 91.8 $\pm$ 0.4   |
|  SYN  | CFQ-mcd2 |  EM / LAS  | Transformer, Ours                                                                                                                            | 80.34 $\pm$ 1.40 / 96.24 $\pm$ 0.68 |
|  SYN  | CFQ-mcd2 |  EM / LAS  | Universal Transformer, [Bergen et al. (2021)](https://proceedings.neurips.cc/paper/2021/hash/0a4dc6dae338c9cb08947c07581f77a2-Abstract.html) |   68.6 $\pm$ 2.3 / 92.5 $\pm$ 0.4   |
|  SYN  | CFQ-mcd2 |  EM / LAS  | Universal Transformer, Ours                                                                                                                  | 78.63 $\pm$ 3.54 / 95.62 $\pm$ 0.75 |
|  SYN  | CFQ-mcd3 |  EM / LAS  | Transformer, [Bergen et al. (2021)](https://proceedings.neurips.cc/paper/2021/hash/0a4dc6dae338c9cb08947c07581f77a2-Abstract.html)           |   48.0 $\pm$ 1.6 / 89.4 $\pm$ 0.3   |
|  SYN  | CFQ-mcd3 |  EM / LAS  | Transformer, Ours                                                                                                                            | 73.43 $\pm$ 6.07 / 94.85 $\pm$ 0.93 |
|  SYN  | CFQ-mcd3 |  EM / LAS  | Universal Transformer, [Bergen et al. (2021)](https://proceedings.neurips.cc/paper/2021/hash/0a4dc6dae338c9cb08947c07581f77a2-Abstract.html) |   59.4 $\pm$ 2.0 / 90.5 $\pm$ 0.5   |
|  SYN  | CFQ-mcd3 |  EM / LAS  | Universal Transformer, Ours                                                                                                                  | 71.49 $\pm$ 5.39 / 94.57 $\pm$ 1.25 |

</details>