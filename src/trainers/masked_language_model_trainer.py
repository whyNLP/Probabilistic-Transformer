import copy
import logging
from pathlib import Path
from typing import List, Union
import time
import datetime
import sys
import inspect
import numpy as np

import torch
from torch.optim.sgd import SGD
from torch.utils.data.dataset import ConcatDataset

from flair.datasets.base import SentenceDataset

try:
    from apex import amp
except ImportError:
    amp = None

import flair
import flair.nn
from flair.datasets import DataLoader
from flair.data import Corpus, FlairDataset, Sentence, Token
from flair.training_utils import (
    init_output_file,
    WeightExtractor,
    log_line,
    add_file_handler,
    Result,
    store_embeddings,
    AnnealOnPlateau,
)
from torch.optim.lr_scheduler import OneCycleLR
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
import random

log = logging.getLogger("flair")


class MaskedLanguageModelTrainer(ModelTrainer):
    def __init__(
        self,
        model: flair.nn.Model,
        corpus: Corpus,
        optimizer: torch.optim.Optimizer = SGD,
        epoch: int = 0,
        mask_rate: float = 0.3,
        use_tensorboard: bool = False,
    ):
        """
        Initialize a model trainer
        :param model: The model that you want to train. The model should inherit from flair.nn.Model
        :param corpus: The dataset used to train the model, should be of type Corpus
        :param optimizer: The optimizer to use (typically SGD or Adam)
        :param epoch: The starting epoch (normally 0 but could be higher if you continue training model)
        :param mask_rate: The rate to mask words.
        :param use_tensorboard: If True, writes out tensorboard information
        """
        self.model: flair.nn.Model = model
        self.corpus: Corpus = corpus
        self.optimizer: torch.optim.Optimizer = optimizer
        self.epoch: int = epoch
        self.mask_rate: float = mask_rate
        self.use_tensorboard: bool = use_tensorboard
    
    def train(
        self,
        base_path: Union[Path, str],
        learning_rate: float = 0.1,
        mini_batch_size: int = 32,
        mini_batch_chunk_size: int = None,
        max_epochs: int = 100,
        scheduler = AnnealOnPlateau,
        pct_start: float = 0.0,
        cycle_momentum: bool = False,
        anneal_factor: float = 0.5,
        patience: int = 3,
        initial_extra_patience = 0,
        min_learning_rate: float = 0.0001,
        train_with_dev: bool = False,
        train_with_test: bool = False,
        monitor_train: bool = False,
        monitor_test: bool = False,
        embeddings_storage_mode: str = "none",
        checkpoint: bool = False,
        save_final_model: bool = True,
        anneal_with_restarts: bool = False,
        anneal_with_prestarts: bool = False,
        batch_growth_annealing: bool = False,
        shuffle: bool = True,
        param_selection_mode: bool = False,
        write_weights: bool = False,
        clip: float = 5.0,
        num_workers: int = 6,
        sampler=None,
        use_amp: bool = False,
        amp_opt_level: str = "O1",
        eval_on_train_fraction=0.0,
        eval_on_train_shuffle=False,
        save_model_at_each_epoch=False,
        fix_train_mask: bool = False,
        fix_dev_mask: bool = False,
        fix_test_mask: bool = False,
        ternary_norm_p=None,
        ternary_norm_lambda=0,
        use_bucket: bool = False,
        **kwargs,
    ) -> dict:
        """
        Trains any class that implements the flair.nn.Model interface.
        :param base_path: Main path to which all output during training is logged and models are saved
        :param learning_rate: Initial learning rate (or max, if scheduler is OneCycleLR)
        :param mini_batch_size: Size of mini-batches during training
        :param mini_batch_chunk_size: If mini-batches are larger than this number, they get broken down into chunks of this size for processing purposes
        :param max_epochs: Maximum number of epochs to train. Terminates training if this number is surpassed.
        :param scheduler: The learning rate scheduler to use
        :param pct_start: Warm up for 'pct_start'(percentage) of the total training time
        :param cycle_momentum: If scheduler is OneCycleLR, whether the scheduler should cycle also the momentum
        :param anneal_factor: The factor by which the learning rate is annealed
        :param patience: Patience is the number of epochs with no improvement the Trainer waits
         until annealing the learning rate
        :param min_learning_rate: If the learning rate falls below this threshold, training terminates
        :param train_with_dev: If True, training is performed using both train+dev data
        :param monitor_train: If True, training data is evaluated at end of each epoch
        :param monitor_test: If True, test data is evaluated at end of each epoch
        :param embeddings_storage_mode: One of 'none' (all embeddings are deleted and freshly recomputed),
        'cpu' (embeddings are stored on CPU) or 'gpu' (embeddings are stored on GPU)
        :param checkpoint: If True, a full checkpoint is saved at end of each epoch
        :param save_final_model: If True, final model is saved
        :param anneal_with_restarts: If True, the last best model is restored when annealing the learning rate
        :param shuffle: If True, data is shuffled during training
        :param param_selection_mode: If True, testing is performed against dev data. Use this mode when doing
        parameter selection.
        :param num_workers: Number of workers in your data loader.
        :param sampler: You can pass a data sampler here for special sampling of data.
        :param eval_on_train_fraction: the fraction of train data to do the evaluation on,
        if 0. the evaluation is not performed on fraction of training data,
        if 'dev' the size is determined from dev set size
        :param eval_on_train_shuffle: if True the train data fraction is determined on the start of training
        and kept fixed during training, otherwise it's sampled at beginning of each epoch
        :param save_model_at_each_epoch: If True, at each epoch the thus far trained model will be saved
        :param kwargs: Other arguments for the Optimizer
        :param fix_train_mask: Fix the random generated mask for training set, so that they are the same for 
        all epochs.
        :param use_bucket: Gather sentence with the same length, to obtain better training speed.
        :return:
        """

        if self.use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                writer = SummaryWriter()
            except:
                log_line(log)
                log.warning(
                    "ATTENTION! PyTorch >= 1.1.0 and pillow are required for TensorBoard support!"
                )
                log_line(log)
                self.use_tensorboard = False
                pass

        if use_amp:
            if sys.version_info < (3, 0):
                raise RuntimeError("Apex currently only supports Python 3. Aborting.")
            if amp is None:
                raise RuntimeError(
                    "Failed to import apex. Please install apex from https://www.github.com/nvidia/apex "
                    "to enable mixed-precision training."
                )

        if mini_batch_chunk_size is None:
            mini_batch_chunk_size = mini_batch_size
        if learning_rate < min_learning_rate:
            min_learning_rate = learning_rate / 10

        initial_learning_rate = learning_rate

        # cast string to Path
        if type(base_path) is str:
            base_path = Path(base_path)

        log_handler = add_file_handler(log, base_path / "training.log")

        log_line(log)
        log.info(f'Model: "{self.model}"')
        log_line(log)
        log.info(f'Corpus: "{self.corpus}"')
        log_line(log)
        log.info("Parameters:")
        log.info(f' - learning_rate: "{learning_rate}"')
        log.info(f' - mini_batch_size: "{mini_batch_size}"')
        log.info(f' - patience: "{patience}"')
        log.info(f' - anneal_factor: "{anneal_factor}"')
        log.info(f' - max_epochs: "{max_epochs}"')
        log.info(f' - shuffle: "{shuffle}"')
        log.info(f' - train_with_dev: "{train_with_dev}"')
        log.info(f' - batch_growth_annealing: "{batch_growth_annealing}"')
        log_line(log)
        log.info(f'Model training base path: "{base_path}"')
        log_line(log)
        log.info(f"Device: {flair.device}")
        log_line(log)
        log.info(f"Embeddings storage mode: {embeddings_storage_mode}")
        if isinstance(self.model, SequenceTagger) and self.model.weight_dict and self.model.use_crf:
            log_line(log)
            log.warning(f'WARNING: Specified class weights will not take effect when using CRF')

        # determine what splits (train, dev, test) to evaluate and log
        log_train = True if monitor_train else False
        log_test = (
            True
            if (not param_selection_mode and self.corpus.test and monitor_test)
            else False
        )
        log_dev = False if train_with_dev or not self.corpus.dev else True
        log_train_part = (
            True
            if (eval_on_train_fraction == "dev" or eval_on_train_fraction > 0.0)
            else False
        )

        if log_train_part:
            train_part_size = (
                len(self.corpus.dev)
                if eval_on_train_fraction == "dev"
                else int(len(self.corpus.train) * eval_on_train_fraction)
            )
            assert train_part_size > 0
            if not eval_on_train_shuffle:
                train_part_indices = list(range(train_part_size))
                train_part = torch.utils.data.dataset.Subset(
                    self.corpus.train, train_part_indices
                )

        # prepare loss logging file and set up header
        loss_txt = init_output_file(base_path, "loss.tsv")

        weight_extractor = WeightExtractor(base_path)

        optimizer: torch.optim.Optimizer = self.optimizer(
            self.model.parameters(), lr=learning_rate, **kwargs
        )

        if use_amp:
            self.model, optimizer = amp.initialize(
                self.model, optimizer, opt_level=amp_opt_level
            )

        # minimize training loss if training with dev data, else maximize dev score
        anneal_mode = "min" if train_with_dev else "max"
        
        if scheduler == OneCycleLR:
            dataset_size = len(self.corpus.train)
            if train_with_dev:
                dataset_size += len(self.corpus.dev)
            lr_scheduler = OneCycleLR(optimizer,
                                   max_lr=learning_rate,
                                   steps_per_epoch=dataset_size//mini_batch_size + 1,
                                   epochs=max_epochs-self.epoch, # if we load a checkpoint, we have already trained for self.epoch
                                   pct_start=pct_start,
                                   cycle_momentum=cycle_momentum)
        else:
            lr_scheduler = scheduler(
                optimizer,
                factor=anneal_factor,
                patience=patience,
                initial_extra_patience=initial_extra_patience,
                mode=anneal_mode,
                verbose=True,
            )
        
        if (isinstance(lr_scheduler, OneCycleLR) and batch_growth_annealing):
            raise ValueError("Batch growth with OneCycle policy is not implemented.")

        train_data = self.corpus.train

        # if training also uses dev/train data, include in training set
        if train_with_dev or train_with_test:

            parts = [self.corpus.train]
            if train_with_dev: parts.append(self.corpus.dev)
            if train_with_test: parts.append(self.corpus.test)

            train_data = ConcatDataset(parts)

        # initialize sampler if provided
        if sampler is not None:
            # init with default values if only class is provided
            if inspect.isclass(sampler):
                sampler = sampler()
            # set dataset to sample from
            sampler.set_dataset(train_data)
            shuffle = False

        dev_score_history = []
        dev_loss_history = []
        train_loss_history = []

        micro_batch_size = mini_batch_chunk_size

        # At any point you can hit Ctrl + C to break out of training early.
        try:
            previous_learning_rate = learning_rate
            momentum = 0
            for group in optimizer.param_groups:
                if "momentum" in group:
                    momentum = group["momentum"]

            for self.epoch in range(self.epoch + 1, max_epochs + 1):
                log_line(log)

                if anneal_with_prestarts:
                    last_epoch_model_state_dict = copy.deepcopy(self.model.state_dict())

                if eval_on_train_shuffle:
                    train_part_indices = list(range(self.corpus.train))
                    random.shuffle(train_part_indices)
                    train_part_indices = train_part_indices[:train_part_size]
                    train_part = torch.utils.data.dataset.Subset(
                        self.corpus.train, train_part_indices
                    )

                # get new learning rate
                for group in optimizer.param_groups:
                    learning_rate = group["lr"]

                if learning_rate != previous_learning_rate and batch_growth_annealing:
                    mini_batch_size *= 2

                # reload last best model if annealing with restarts is enabled
                if (
                    (anneal_with_restarts or anneal_with_prestarts)
                    and learning_rate != previous_learning_rate
                    and (base_path / "best-model.pt").exists()
                ):
                    if anneal_with_restarts:
                        log.info("resetting to best model")
                        self.model.load_state_dict(
                            self.model.load(base_path / "best-model.pt").state_dict()
                        )
                    if anneal_with_prestarts:
                        log.info("resetting to pre-best model")
                        self.model.load_state_dict(
                            self.model.load(base_path / "pre-best-model.pt").state_dict()
                        )

                previous_learning_rate = learning_rate

                # stop training if learning rate becomes too small
                if (not isinstance(lr_scheduler, OneCycleLR)) and learning_rate < min_learning_rate:
                    log_line(log)
                    log.info("learning rate too small - quitting training!")
                    log_line(log)
                    break

                batch_loader = DataLoader(
                    self.mask_dataset(
                        train_data, 
                        fix_random=fix_train_mask, 
                        use_bucket=use_bucket if self.epoch > 1 else False, 
                        shuffle=shuffle if self.epoch > 1 else False, 
                        micro_batch_size=micro_batch_size
                    ),
                    batch_size=mini_batch_size,
                    shuffle=shuffle if (self.epoch > 1 and not use_bucket) else False, # never shuffle the first epoch
                    num_workers=num_workers,
                    sampler=sampler,
                )

                self.model.train()

                train_loss: float = 0

                seen_batches = 0
                total_number_of_batches = len(batch_loader)

                modulo = max(1, int(total_number_of_batches / 10))

                # process mini-batches
                batch_time = 0
                for batch_no, batch in enumerate(batch_loader):

                    start_time = time.time()

                    # zero the gradients on the model and optimizer
                    self.model.zero_grad()
                    optimizer.zero_grad()

                    # if necessary, make batch_steps
                    batch_steps = [batch]
                    if len(batch) > micro_batch_size:
                        batch_steps = [
                            batch[x : x + micro_batch_size]
                            for x in range(0, len(batch), micro_batch_size)
                        ]

                    # forward and backward for batch
                    for batch_step in batch_steps:

                        # forward pass
                        loss = self.model.forward_loss(batch_step)

                        # add norm
                        if ternary_norm_p is not None:
                            loss += ternary_norm_lambda * self.model.mod.getTernaryNorm(ternary_norm_p)

                        # Backward
                        if use_amp:
                            with amp.scale_loss(loss, optimizer) as scaled_loss:
                                scaled_loss.backward()
                        else:
                            loss.backward()

                    # do the optimizer step
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
                    optimizer.step()
                    
                    # do the scheduler step if one-cycle
                    if isinstance(lr_scheduler, OneCycleLR):
                        lr_scheduler.step()
                        # get new learning rate
                        for group in optimizer.param_groups:
                            learning_rate = group["lr"]
                            if "momentum" in group:
                                momentum = group["momentum"]                    

                    seen_batches += 1
                    train_loss += loss.item()

                    # depending on memory mode, embeddings are moved to CPU, GPU or deleted
                    store_embeddings(batch, embeddings_storage_mode)

                    batch_time += time.time() - start_time
                    if seen_batches % modulo == 0:
                        momentum_info = f' - momentum: {momentum:.4f}' if cycle_momentum else ''
                        log.info(
                            f"epoch {self.epoch} - iter {seen_batches}/{total_number_of_batches} - loss "
                            f"{train_loss / seen_batches:.8f} - samples/sec: {mini_batch_size * modulo / batch_time:.2f}"
                            f" - lr: {learning_rate:.6f}{momentum_info}"
                        )
                        batch_time = 0
                        iteration = self.epoch * total_number_of_batches + batch_no
                        if not param_selection_mode and write_weights:
                            weight_extractor.extract_weights(
                                self.model.state_dict(), iteration
                            )

                train_loss /= seen_batches

                self.model.eval()

                log_line(log)
                log.info(
                    f"EPOCH {self.epoch} done: loss {train_loss:.4f} - lr {learning_rate:.7f}"
                )

                if self.use_tensorboard:
                    writer.add_scalar("train_loss", train_loss, self.epoch)

                # anneal against train loss if training with dev, otherwise anneal against dev score
                current_score = train_loss

                # evaluate on train / dev / test split depending on training settings
                result_line: str = ""

                if log_train:
                    train_eval_result, train_loss = self.model.evaluate(
                        self.mask_dataset(self.corpus.train, fix_random=fix_train_mask, use_bucket=use_bucket),
                        mini_batch_size=mini_batch_chunk_size,
                        num_workers=num_workers,
                        embedding_storage_mode=embeddings_storage_mode,
                    )
                    result_line += f"\t{train_eval_result.log_line}"

                    # depending on memory mode, embeddings are moved to CPU, GPU or deleted
                    store_embeddings(self.corpus.train, embeddings_storage_mode)

                if log_train_part:
                    train_part_eval_result, train_part_loss = self.model.evaluate(
                        self.mask_dataset(train_part, fix_random=fix_train_mask, use_bucket=use_bucket),
                        mini_batch_size=mini_batch_chunk_size,
                        num_workers=num_workers,
                        embedding_storage_mode=embeddings_storage_mode,
                    )
                    result_line += (
                        f"\t{train_part_loss}\t{train_part_eval_result.log_line}"
                    )
                    log.info(
                        f"TRAIN_SPLIT : loss {train_part_loss} - score {round(train_part_eval_result.main_score, 4)}"
                    )

                if log_dev:
                    dev_eval_result, dev_loss = self.model.evaluate(
                        self.mask_dataset(self.corpus.dev, fix_random=fix_dev_mask, use_bucket=use_bucket),
                        mini_batch_size=mini_batch_chunk_size,
                        num_workers=num_workers,
                        out_path=base_path / "dev.tsv",
                        embedding_storage_mode=embeddings_storage_mode,
                    )
                    result_line += f"\t{dev_loss}\t{dev_eval_result.log_line}"
                    log.info(
                        f"DEV : loss {dev_loss} - score {round(dev_eval_result.main_score, 4)}"
                    )
                    # calculate scores using dev data if available
                    # append dev score to score history
                    dev_score_history.append(dev_eval_result.main_score)
                    dev_loss_history.append(dev_loss.item())

                    current_score = dev_eval_result.main_score

                    # depending on memory mode, embeddings are moved to CPU, GPU or deleted
                    store_embeddings(self.corpus.dev, embeddings_storage_mode)

                    if self.use_tensorboard:
                        writer.add_scalar("dev_loss", dev_loss, self.epoch)
                        writer.add_scalar(
                            "dev_score", dev_eval_result.main_score, self.epoch
                        )

                if log_test:
                    test_eval_result, test_loss = self.model.evaluate(
                        self.mask_dataset(self.corpus.test, fix_random=fix_test_mask, use_bucket=use_bucket),
                        mini_batch_size=mini_batch_chunk_size,
                        num_workers=num_workers,
                        out_path=base_path / "test.tsv",
                        embedding_storage_mode=embeddings_storage_mode,
                    )
                    result_line += f"\t{test_loss}\t{test_eval_result.log_line}"
                    log.info(
                        f"TEST : loss {test_loss} - score {round(test_eval_result.main_score, 4)}"
                    )

                    # depending on memory mode, embeddings are moved to CPU, GPU or deleted
                    store_embeddings(self.corpus.test, embeddings_storage_mode)

                    if self.use_tensorboard:
                        writer.add_scalar("test_loss", test_loss, self.epoch)
                        writer.add_scalar(
                            "test_score", test_eval_result.main_score, self.epoch
                        )

                # determine learning rate annealing through scheduler. Use auxiliary metric for AnnealOnPlateau
                if log_dev and isinstance(lr_scheduler, AnnealOnPlateau):
                    lr_scheduler.step(current_score, dev_loss)
                elif not isinstance(lr_scheduler, OneCycleLR):
                    lr_scheduler.step(current_score)
                elif log_dev and isinstance(lr_scheduler, OneCycleLR):
                    if hasattr(lr_scheduler, "score_history"):
                        lr_scheduler.score_history.append((current_score, dev_loss))
                        if current_score > lr_scheduler.best:
                            lr_scheduler.num_bad_epochs = 0
                            lr_scheduler.best = current_score
                        else:
                            lr_scheduler.num_bad_epochs += 1
                    else:
                        lr_scheduler.score_history = [(current_score, dev_loss)]
                        lr_scheduler.best = current_score
                        lr_scheduler.num_bad_epochs = 0

                train_loss_history.append(train_loss)

                # determine bad epoch number
                try:
                    bad_epochs = lr_scheduler.num_bad_epochs
                except:
                    bad_epochs = 0
                for group in optimizer.param_groups:
                    new_learning_rate = group["lr"]
                if not isinstance(lr_scheduler, OneCycleLR) and new_learning_rate != previous_learning_rate:
                    bad_epochs = patience + 1
                    if previous_learning_rate == initial_learning_rate: bad_epochs += initial_extra_patience

                # log bad epochs
                log.info(f"BAD EPOCHS (no improvement): {bad_epochs}")

                # output log file
                with open(loss_txt, "a") as f:

                    # make headers on first epoch
                    if self.epoch == 1:
                        f.write(
                            f"EPOCH\tTIMESTAMP\tBAD_EPOCHS\tLEARNING_RATE\tTRAIN_LOSS"
                        )

                        if log_train:
                            f.write(
                                "\tTRAIN_"
                                + "\tTRAIN_".join(
                                    train_eval_result.log_header.split("\t")
                                )
                            )
                        if log_train_part:
                            f.write(
                                "\tTRAIN_PART_LOSS\tTRAIN_PART_"
                                + "\tTRAIN_PART_".join(
                                    train_part_eval_result.log_header.split("\t")
                                )
                            )
                        if log_dev:
                            f.write(
                                "\tDEV_LOSS\tDEV_"
                                + "\tDEV_".join(dev_eval_result.log_header.split("\t"))
                            )
                        if log_test:
                            f.write(
                                "\tTEST_LOSS\tTEST_"
                                + "\tTEST_".join(
                                    test_eval_result.log_header.split("\t")
                                )
                            )

                    f.write(
                        f"\n{self.epoch}\t{datetime.datetime.now():%H:%M:%S}\t{bad_epochs}\t{learning_rate:.4f}\t{train_loss}"
                    )
                    f.write(result_line)

                # if checkpoint is enabled, save model at each epoch
                if checkpoint and not param_selection_mode:
                    self.save_checkpoint(base_path / "checkpoint.pt")

                # if we use dev data, remember best model based on dev evaluation score
                if (
                    (not train_with_dev or anneal_with_restarts or anneal_with_prestarts)
                    and not param_selection_mode
                    and not isinstance(lr_scheduler, OneCycleLR)
                    and current_score == lr_scheduler.best
                    and bad_epochs == 0
                ) or (
                    (not train_with_dev or anneal_with_restarts or anneal_with_prestarts)
                    and not param_selection_mode
                    and isinstance(lr_scheduler, OneCycleLR)
                    and current_score == lr_scheduler.best
                    and bad_epochs == 0
                ):
                    print("saving best model")
                    self.model.save(base_path / "best-model.pt")

                    if anneal_with_prestarts:
                        current_state_dict = self.model.state_dict()
                        self.model.load_state_dict(last_epoch_model_state_dict)
                        self.model.save(base_path / "pre-best-model.pt")
                        self.model.load_state_dict(current_state_dict)
                        
                if save_model_at_each_epoch:
                    print("saving model of current epoch")
                    model_name = "model_epoch_" + str(self.epoch) + ".pt"
                    self.model.save(base_path / model_name)

            # if we do not use dev data for model selection, save final model
            if save_final_model and not param_selection_mode:
                self.model.save(base_path / "final-model.pt")

        except KeyboardInterrupt:
            log_line(log)
            log.info("Exiting from training early.")

            if self.use_tensorboard:
                writer.close()

            if not param_selection_mode:
                log.info("Saving model ...")
                self.model.save(base_path / "final-model.pt")
                log.info("Done.")

        # test best model if test data is present
        if self.corpus.test and not train_with_test:
            final_score = self.final_test(base_path, mini_batch_chunk_size, num_workers)
        else:
            final_score = 0
            log.info("Test data not provided setting final score to 0")

        log.removeHandler(log_handler)

        if self.use_tensorboard:
            writer.close()

        return {
            "test_score": final_score,
            "dev_score_history": dev_score_history,
            "train_loss_history": train_loss_history,
            "dev_loss_history": dev_loss_history,
        }

    def mask_dataset(self, dataset: FlairDataset, fix_random: bool = False, use_bucket=False, shuffle=False, micro_batch_size=16):
        if fix_random:
            rand = random.Random(313) # My office is at 1A-313, SIST, ShanghaiTech University. A lucky number.
        else:
            rand = random
        return SentenceDataset(self.mask_sentences(dataset, rand, use_bucket, shuffle, micro_batch_size))

    def mask_sentences(self, sentences: List[Sentence], rand=None, use_bucket=False, shuffle=False, micro_batch_size=16):
        """
        Return a list of brand new sentences with masks. 'mlm' tags indicate the mask status:

        Original:     I    went    to     <MASK>   yesterday    .
        'mlm' tag:  <pad>  <pad>  <pad>  Shanghai    <pad>    <pad>
        """
        masked_sentences = [self.mask_sentence(sentence, rand) for sentence in sentences]
        if use_bucket:
            if shuffle:
                masked_sentences.sort(key = lambda x: (len(x), random.random()))
                grouped = [masked_sentences[i:i+micro_batch_size] for i in range(0,len(masked_sentences),micro_batch_size)]
                grouped_chunked, last_group = grouped[:-1], grouped[-1]
                random.shuffle(grouped_chunked)
                masked_sentences = [x for l in grouped_chunked for x in l] + last_group
            else:
                masked_sentences.sort(key = lambda x: len(x))

        return masked_sentences

    def mask_sentence(self, sentence: Sentence, rand=None):
        try: # accelerate by calculating tag idx in advance
            new_sentence = Sentence()
            tag_idx = []
            for token in sentence:
                if flip_coin(self.mask_rate, rand):
                    new_token = Token("<MASK>")
                    new_token.add_tag('mlm', token.text)
                    tag_idx.append(self.model.tag_dictionary.get_idx_for_item(token.text))
                else:
                    new_token = Token(text = token.text)
                    new_token.add_tag('mlm', '<pad>')
                    tag_idx.append(-100)
                new_sentence.add_token(new_token)
            new_sentence.tag_idx = torch.tensor(tag_idx, device=flair.device)
        except Exception:
            new_sentence = Sentence()
            tag_idx = []
            for token in sentence:
                if flip_coin(self.mask_rate, rand):
                    new_token = Token("<MASK>")
                    new_token.add_tag('mlm', token.text)
                else:
                    new_token = Token(text = token.text)
                    new_token.add_tag('mlm', '<pad>')
                new_sentence.add_token(new_token)
        return new_sentence
    
    def final_test(
        self, base_path: Union[Path, str], eval_mini_batch_size: int, num_workers: int = 8
    ):
        if type(base_path) is str:
            base_path = Path(base_path)

        log.info('-' * 100)
        log.info("Testing using best model ...")

        self.model.eval()

        if (base_path / "best-model.pt").exists():
            self.model = self.model.load(base_path / "best-model.pt")
        
        test_results, test_loss = self.model.evaluate(
            self.mask_dataset(self.corpus.test, fix_random=True),
            mini_batch_size=eval_mini_batch_size,
            num_workers=num_workers,
            out_path=base_path / "test.tsv",
            embedding_storage_mode="none",
        )

        log.info(test_results.log_header + '\t' + test_results.log_line)
        log.info(test_results.detailed_results)
        log.info('-' * 100)

        # get and return the final test score of best model
        final_score = test_results.main_score

        return final_score


def flip_coin(p: float, rand=None):
    if rand is None:
        rand = random
    return rand.random() < p
        