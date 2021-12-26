import time, datetime
import random
import sys
from pathlib import Path
from typing import Union

from torch import cuda
from torch.utils.data import Dataset, DataLoader
from torch.optim.sgd import SGD

try:
    from apex import amp
except ImportError:
    amp = None

import flair
from flair.data import Dictionary, Corpus
from flair.models import SequenceTagger
from flair.optim import *
from flair.training_utils import add_file_handler

log = logging.getLogger("flair")


class MaskedLanguageModelTrainer:
    def __init__(
        self,
        model: SequenceTagger,
        corpus: Corpus,
        optimizer: Optimizer = SGD,
        test_mode: bool = False,
        epoch: int = 0,
        loss: float = 10000,
        optimizer_state: dict = None,
    ):
        self.model: SequenceTagger = model
        self.optimizer: Optimizer = optimizer
        self.corpus: Corpus = corpus
        self.test_mode: bool = test_mode

        self.loss_function = torch.nn.CrossEntropyLoss()
        self.log_interval = 100
        self.epoch = epoch
        self.loss = loss
        self.optimizer_state = optimizer_state

    def train(
        self,
        base_path: Union[Path, str],
        learning_rate: float = 20,
        mini_batch_size: int = 100,
        mini_batch_chunk_size: int = None,
        anneal_factor: float = 0.25,
        patience: int = 10,
        clip=0.25,
        max_epochs: int = 1000,
        checkpoint: bool = False,
        save_model_at_each_epoch=False,
        num_workers: int = 2,
        use_amp: bool = False,
        amp_opt_level: str = "O1",
        **kwargs,
    ):

        if use_amp:
            if sys.version_info < (3, 0):
                raise RuntimeError("Apex currently only supports Python 3. Aborting.")
            if amp is None:
                raise RuntimeError(
                    "Failed to import apex. Please install apex from https://www.github.com/nvidia/apex "
                    "to enable mixed-precision training."
                )

        # cast string to Path
        if type(base_path) is str:
            base_path = Path(base_path)

        add_file_handler(log, base_path / "training.log")

        base_path.mkdir(parents=True, exist_ok=True)
        loss_txt = base_path / "loss.txt"
        savefile = base_path / "best-lm.pt"

        if mini_batch_chunk_size is None:
            mini_batch_chunk_size = mini_batch_size

        try:
            best_val_loss = self.loss
            optimizer = self.optimizer(
                self.model.parameters(), lr=learning_rate, **kwargs
            )
            if self.optimizer_state is not None:
                optimizer.load_state_dict(self.optimizer_state)

            if isinstance(optimizer, (AdamW, SGDW)):
                scheduler: ReduceLRWDOnPlateau = ReduceLRWDOnPlateau(
                    optimizer, verbose=True, factor=anneal_factor, patience=patience
                )
            else:
                scheduler: ReduceLROnPlateau = ReduceLROnPlateau(
                    optimizer, verbose=True, factor=anneal_factor, patience=patience
                )

            if use_amp:
                self.model, optimizer = amp.initialize(
                    self.model, optimizer, opt_level=amp_opt_level
                )

            training_generator = DataLoader(
                self.corpus.train, shuffle=False, num_workers=num_workers
            )

            for epoch in range(self.epoch, max_epochs):
                epoch_start_time = time.time()
                # Shuffle training files randomly after serially iterating through corpus one
                batch_loader = DataLoader(
                    train_data,
                    batch_size=mini_batch_size,
                    shuffle=True if self.epoch > 1 else False, # never shuffle the first epoch
                    num_workers=num_workers
                )

                if epoch > 0 and save_model_at_each_epoch:
                    print("saving model of current epoch")
                    self.model.save_checkpoint(
                        base_path / f"epoch_{epoch}.pt",
                        optimizer,
                        epoch,
                        0,
                        best_val_loss,
                    )

                # process mini-batches
                batch_time = 0
                for batch_no, batch in enumerate(batch_loader):

                    start_time = time.time()

                    # zero the gradients on the model and optimizer
                    self.model.zero_grad()
                    optimizer.zero_grad()

                    # if necessary, make batch_steps
                    batch_steps = [batch]
                    if len(batch) > mini_batch_chunk_size:
                        batch_steps = [
                            batch[x : x + mini_batch_chunk_size]
                            for x in range(0, len(batch), mini_batch_chunk_size)
                        ]

                    # forward and backward for batch
                    for batch_step in batch_steps:

                        # forward pass
                        loss = self.model.forward_loss(batch_step)

                        # Backward
                        if use_amp:
                            with amp.scale_loss(loss, optimizer) as scaled_loss:
                                scaled_loss.backward()
                        else:
                            loss.backward()

                    # do the optimizer step
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
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

                # iterate through training data
                for curr_batch_num, train_data in enumerate(batch_loader):

                    split_start_time = time.time()
                    # off by one for printing
                    curr_batch_num += 1

                    log.info(
                        "Batch %d" % curr_batch_num
                        + "\t - ({:%H:%M:%S})".format(datetime.datetime.now())
                    )

                    for group in optimizer.param_groups:
                        learning_rate = group["lr"]

                    # go into train mode
                    self.model.train()

                    # reset variables
                    hidden = self.model.init_hidden(mini_batch_size)

                    # not really sure what this does
                    ntokens = len(self.corpus.dictionary)

                    total_loss = 0
                    start_time = time.time()

                    for batch, i in enumerate(
                        range(0, train_data.size(0) - 1, sequence_length)
                    ):
                        data, targets = self._get_batch(train_data, i, sequence_length)

                        if not data.is_cuda and cuda.is_available():
                            log.info(
                                "Batch %d is not on CUDA, training will be very slow"
                                % (batch)
                            )
                            raise Exception("data isnt on cuda")

                        self.model.zero_grad()
                        optimizer.zero_grad()

                        # do the forward pass in the model
                        output, rnn_output, hidden = self.model.forward(data, hidden)

                        # try to predict the targets
                        loss = self.loss_function(output.view(-1, ntokens), targets)
                        # Backward
                        if use_amp:
                            with amp.scale_loss(loss, optimizer) as scaled_loss:
                                scaled_loss.backward()
                        else:
                            loss.backward()

                        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)

                        optimizer.step()

                        total_loss += loss.data

                        # We detach the hidden state from how it was previously produced.
                        # If we didn't, the model would try backpropagating all the way to start of the dataset.
                        hidden = self._repackage_hidden(hidden)

                        # explicitly remove loss to clear up memory
                        del loss, output, rnn_output

                        if batch % self.log_interval == 0 and batch > 0:
                            cur_loss = total_loss.item() / self.log_interval
                            elapsed = time.time() - start_time
                            log.info(
                                "| split {:3d} /{:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | "
                                "loss {:5.2f} | ppl {:8.2f}".format(
                                    curr_batch_num,
                                    number_of_splits,
                                    batch,
                                    len(train_data) // sequence_length,
                                    elapsed * 1000 / self.log_interval,
                                    cur_loss,
                                    math.exp(cur_loss),
                                )
                            )
                            total_loss = 0
                            start_time = time.time()

                    log.info(
                        "%d seconds for train split %d"
                        % (time.time() - split_start_time, curr_batch_num)
                    )

                    ###############################################################################
                    self.model.eval()

                    val_loss = self.evaluate(val_data, mini_batch_size, sequence_length)
                    scheduler.step(val_loss)

                    log.info("best loss so far {:5.2f}".format(best_val_loss))

                    log.info(self.model.generate_text())

                    if checkpoint:
                        self.model.save_checkpoint(
                            base_path / "checkpoint.pt",
                            optimizer,
                            epoch,
                            curr_batch_num,
                            best_val_loss,
                        )

                    # Save the model if the validation loss is the best we've seen so far.
                    if val_loss < best_val_loss:
                        self.model.best_score = best_val_loss
                        self.model.save(savefile)
                        best_val_loss = val_loss

                    ###############################################################################
                    # print info
                    ###############################################################################
                    log.info("-" * 89)

                    summary = (
                        "| end of split {:3d} /{:3d} | epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | "
                        "valid ppl {:8.2f} | learning rate {:3.4f}".format(
                            curr_batch_num,
                            number_of_splits,
                            epoch + 1,
                            (time.time() - split_start_time),
                            val_loss,
                            math.exp(val_loss),
                            learning_rate,
                        )
                    )

                    with open(loss_txt, "a") as myfile:
                        myfile.write("%s\n" % summary)

                    log.info(summary)
                    log.info("-" * 89)

                log.info("Epoch time: %.2f" % (time.time() - epoch_start_time))

        except KeyboardInterrupt:
            log.info("-" * 89)
            log.info("Exiting from training early")

        ###############################################################################
        # final testing
        ###############################################################################
        test_data = self._batchify(self.corpus.test, mini_batch_size)
        test_loss = self.evaluate(test_data, mini_batch_size, sequence_length)

        summary = "TEST: valid loss {:5.2f} | valid ppl {:8.2f}".format(
            test_loss, math.exp(test_loss)
        )
        with open(loss_txt, "a") as myfile:
            myfile.write("%s\n" % summary)

        log.info(summary)
        log.info("-" * 89)

    def evaluate(self, data_source, eval_batch_size, sequence_length):
        # Turn on evaluation mode which disables dropout.
        self.model.eval()

        with torch.no_grad():
            total_loss = 0
            ntokens = len(self.corpus.dictionary)

            hidden = self.model.init_hidden(eval_batch_size)

            for i in range(0, data_source.size(0) - 1, sequence_length):
                data, targets = self._get_batch(data_source, i, sequence_length)
                prediction, rnn_output, hidden = self.model.forward(data, hidden)
                output_flat = prediction.view(-1, ntokens)
                total_loss += len(data) * self.loss_function(output_flat, targets).data
                hidden = self._repackage_hidden(hidden)
            return total_loss.item() / len(data_source)

    @staticmethod
    def _batchify(data, batch_size):
        # Work out how cleanly we can divide the dataset into bsz parts.
        nbatch = data.size(0) // batch_size
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = data.narrow(0, 0, nbatch * batch_size)
        # Evenly divide the data across the bsz batches.
        data = data.view(batch_size, -1).t().contiguous()
        return data

    @staticmethod
    def _get_batch(source, i, sequence_length):
        seq_len = min(sequence_length, len(source) - 1 - i)

        data = source[i : i + seq_len].clone().detach()
        target = source[i + 1 : i + 1 + seq_len].view(-1).clone().detach()

        data = data.to(flair.device)
        target = target.to(flair.device)

        return data, target

    @staticmethod
    def _repackage_hidden(h):
        """Wraps hidden states in new tensors, to detach them from their history."""
        return tuple(v.clone().detach() for v in h)

    @staticmethod
    def load_from_checkpoint(
        checkpoint_file: Union[str, Path], corpus: TextCorpus, optimizer: Optimizer = SGD
    ):
        if type(checkpoint_file) is str:
            checkpoint_file = Path(checkpoint_file)

        checkpoint = LanguageModel.load_checkpoint(checkpoint_file)
        return LanguageModelTrainer(
            checkpoint["model"],
            corpus,
            optimizer,
            epoch=checkpoint["epoch"],
            split=checkpoint["split"],
            loss=checkpoint["loss"],
            optimizer_state=checkpoint["optimizer_state_dict"],
        )
