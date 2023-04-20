#!/usr/bin/env python3
import sys
import os
import torch
import logging
import math
import speechbrain as sb
import torchaudio
from hyperpyyaml import load_hyperpyyaml
from datasets import load_metric
from transformers import Wav2Vec2CTCTokenizer
import random
from pyctcdecode import build_ctcdecoder
from step_counter import StepCounter
from tqdm import tqdm

"""Recipe for training a sequence-to-sequence ASR system with CommonVoice.
The system employs a wav2vec2 encoder and a CTC decoder.
Decoding is performed with greedy decoding (will be extended to beam search).

To run this recipe, do the following:
> python train_with_wav2vec2.py hparams/train_with_wav2vec2.yaml

With the default hyperparameters, the system employs a pretrained wav2vec2 encoder.
The wav2vec2 model is pretrained following the model given in the hprams file.
It may be dependent on the language.

The neural network is trained with CTC on sub-word units estimated with
Byte Pairwise Encoding (BPE).

The experiment file is flexible enough to support a large variety of
different systems. By properly changing the parameter files, you can try
different encoders, decoders, tokens (e.g, characters instead of BPE),
training languages (all CommonVoice languages), and many
other possible variations.

Authors
 * Titouan Parcollet 2021
"""

logger = logging.getLogger(__name__)


def get_decoder_ngram_model(tokenizer, ngram_lm_path, vocab_path=None):
    unigrams = None
    if vocab_path is not None:
        unigrams = []
        with open(vocab_path, encoding='utf-8') as f:
            for line in f:
                unigrams.append(line.strip())

    vocab_dict = tokenizer.get_vocab()
    sort_vocab = sorted((value, key) for (key, value) in vocab_dict.items())
    vocab = [x[1] for x in sort_vocab]
    vocab_list = vocab

    # convert ctc blank character representation
    vocab_list[tokenizer.pad_token_id] = ""
    # replace special characters
    vocab_list[tokenizer.word_delimiter_token_id] = " "
    # specify ctc blank char index, since conventionally it is the last entry of the logit matrix
    decoder = build_ctcdecoder(vocab_list, ngram_lm_path, unigrams=unigrams)
    return decoder

# Define training procedure
class ASR(sb.core.Brain):
    def __init__(self, *args, **kwargs):
        super(ASR, self).__init__(*args, **kwargs)
        self.step_counter = StepCounter()
        self.gradient_accumulation = self.hparams.gradient_accumulation
        self.accumulation_size = self.hparams.accumulation_sec # * self.hparams.sample_rate
        self.num_batch = -1
        self.valid_set = None

    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""

        wavs, wav_lens = batch.sig

        if stage == sb.Stage.TRAIN:
            # Augmentation
            if hasattr(self.hparams, "env_corrupt") and random.random() < self.hparams.corrupt_prob:
                wavs = self.hparams.env_corrupt(wavs, wav_lens)

            if hasattr(self.hparams, "augmentation"):
                wavs = self.hparams.augmentation(wavs)

        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)

        if self.hparams.normalize_wav:
            with torch.no_grad():
                wavs = torch.nn.functional.layer_norm(wavs, wavs.shape[1:], eps=1e-7)

        # Forward pass
        feats = self.modules.wav2vec2(wavs)
        x = self.modules.dropout(feats)
        logits = self.modules.lm_head(x)

        return logits, wav_lens

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss (CTC) given predictions and targets."""
        logits, wav_lens = predictions

        ids = batch.id
        tokens, tokens_lens = batch.tokens
        tokens, tokens_lens = tokens.to(self.device), tokens_lens.to(self.device)

        p_ctc = self.hparams.log_softmax(logits)
        loss = self.hparams.ctc_cost(p_ctc, tokens, wav_lens, tokens_lens)

        if stage != sb.Stage.TRAIN:
            if stage == sb.Stage.TEST and self.hparams.use_lm:
                predicted_words = [self.hparams.lm.decode(logit.detach().cpu().numpy(), beam_width=500) for logit in logits]
            else:
                # Decode token terms to words
                sequence = sb.decoders.ctc_greedy_decode(
                    p_ctc, wav_lens, blank_id=self.hparams.blank_index
                )

                predicted_words = self.tokenizer.batch_decode(
                    sequence, skip_special_tokens=True, group_tokens=False)


            # Convert indices to words
            target_words = batch.words

            self.wer_metric.append(ids, predicted_words, target_words)
            self.cer_metric.append(ids, predicted_words, target_words)
            self.hug_wer_metric.add_batch(
                predictions=predicted_words, references=target_words)

        return loss

    def fit_batch(self, batch):
        """Train the parameters given a single batch in input"""
        if self.auto_mix_prec:

            # self.wav2vec_optimizer.zero_grad(set_to_none=True)
            # self.model_optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast():
                outputs = self.compute_forward(batch, sb.Stage.TRAIN)
                loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)

            self.scaler.scale(loss / self.gradient_accumulation).backward()
            if self.step % self.gradient_accumulation == 0 or self.step == self.num_batch:
                self.scaler.unscale_(self.wav2vec_optimizer)
                self.scaler.unscale_(self.model_optimizer)

                if self.check_gradients(loss):
                    self.scaler.step(self.wav2vec_optimizer)
                    self.scaler.step(self.adam_optimizer)

                self.modules.zero_grad(set_to_none=True)
                self.scaler.update()
                self.step_counter.update()
                log_stats = {
                    "step_loss": loss.detach(),
                    "lr": self.model_optimizer.param_groups[0]["lr"],
                    "lr_wav2vec": self.wav2vec_optimizer.param_groups[0]["lr"],
                    "accumulate": self.gradient_accumulation
                }
                self.hparams.train_logger.run.log(
                    log_stats,
                    step=self.step_counter.current
                )

                batch_duration = batch.duration[-1] * self.hparams.batch_size
                if self.step == self.num_batch:
                    self.gradient_accumulation = self.hparams.gradient_accumulation
                else:
                    next_accumulation = math.ceil(self.accumulation_size / batch_duration)
                    if next_accumulation < self.gradient_accumulation:
                        self.gradient_accumulation = next_accumulation
        else:
            outputs = self.compute_forward(batch, sb.Stage.TRAIN)
            loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)
            (loss / self.gradient_accumulation).backward()

            if self.step % self.gradient_accumulation == 0 or self.step == self.num_batch:
                if self.check_gradients(loss):
                    self.wav2vec_optimizer.step()
                    self.model_optimizer.step()

                # self.modules.zero_grad(set_to_none=True)
                self.model_optimizer.zero_grad(set_to_none=True)
                self.wav2vec_optimizer.zero_grad(set_to_none=True)

                self.step_counter.update()
                log_stats = {
                    "step_loss": loss.detach(),
                    "lr": self.model_optimizer.param_groups[0]["lr"],
                    "lr_wav2vec": self.wav2vec_optimizer.param_groups[0]["lr"],
                    "accumulate": self.gradient_accumulation
                }
                self.hparams.train_logger.run.log(
                    log_stats,
                    step=self.step_counter.current
                )

                batch_duration = batch.duration[-1] * self.hparams.batch_size
                if self.step == self.num_batch:
                    self.gradient_accumulation = self.hparams.gradient_accumulation
                elif self.accumulation_size:
                    next_accumulation = math.ceil(self.accumulation_size / batch_duration)
                    if next_accumulation < self.gradient_accumulation:
                        self.gradient_accumulation = next_accumulation

        return loss.detach()

    def evaluate_batch(self, batch, stage):
        """Computations needed for validation/test batches"""
        with torch.no_grad():
            predictions = self.compute_forward(batch, stage=stage)
            loss = self.compute_objectives(predictions, batch, stage=stage)
        return loss.detach()

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        if stage != sb.Stage.TRAIN:
            self.cer_metric = self.hparams.cer_computer()
            self.wer_metric = self.hparams.error_rate_computer()
            self.hug_wer_metric = load_metric('wer')

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of an epoch."""
        # Compute/store important stats
        step = self.step_counter.current
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["CER"] = self.cer_metric.summarize("error_rate")
            stage_stats["WER"] = self.wer_metric.summarize("error_rate")
            stage_stats["wer"] = self.hug_wer_metric.compute() * 100

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            old_lr_model, new_lr_model = self.hparams.lr_annealing_model(
                stage_stats["loss"]
            )
            old_lr_wav2vec, new_lr_wav2vec = self.hparams.lr_annealing_wav2vec(
                stage_stats["loss"]
            )
            sb.nnet.schedulers.update_learning_rate(
                self.model_optimizer, new_lr_model
            )
            sb.nnet.schedulers.update_learning_rate(
                self.wav2vec_optimizer, new_lr_wav2vec
            )
            self.hparams.train_logger.log_stats(
                stats_meta={
                    "epoch": step,
                },
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.hparams.train_logger_txt.log_stats(
                stats_meta={
                    "epoch": epoch,
                    "step": step,
                    "lr_model": old_lr_model,
                },
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"WER": stage_stats["WER"], "epoch": epoch}, min_keys=["WER"],
            )
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger_txt.log_stats(
                stats_meta={
                    "Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            with open(self.hparams.wer_file, "w") as w:
                self.wer_metric.write_stats(w)

    def init_optimizers(self):
        "Initializes the wav2vec2 optimizer and model optimizer"
        self.wav2vec_optimizer = self.hparams.wav2vec_opt_class(
            self.modules.wav2vec2.parameters()
        )
        self.model_optimizer = self.hparams.model_opt_class(
            self.hparams.model.parameters()
        )

        if self.checkpointer is not None:
            self.checkpointer.add_recoverable(
                "wav2vec_opt", self.wav2vec_optimizer
            )
            self.checkpointer.add_recoverable("modelopt", self.model_optimizer)
            self.checkpointer.add_recoverable("update_step", self.step_counter)
    
    def on_fit_start(self):
        """Gets called at the beginning of ``fit()``"""
        super().on_fit_start()
        # self.modules.wav2vec2.model.gradient_checkpointing_disable()
        self.modules.wav2vec2.normalize_wav = False
    
    def fit(
        self,
        epoch_counter,
        train_set,
        valid_set=None,
        progressbar=None,
        train_loader_kwargs={},
        valid_loader_kwargs={},
    ):
        self.valid_set = sb.dataio.dataloader.make_dataloader(valid_set, batch_size=4)
        batch_size = train_loader_kwargs.get('batch_size', 1)
        self.num_batch = math.ceil(len(train_set) / batch_size)
        super().fit(
            epoch_counter,
            train_set,
            valid_set,
            progressbar,
            train_loader_kwargs,
            valid_loader_kwargs,
        )
    
    def _save_intra_epoch_ckpt(self):
        """Saves a CKPT with specific intra-epoch flag."""
        super()._save_intra_epoch_ckpt()
        # self._intra_evaluate()
    
    def _intra_evaluate(self):
        # Validation stage
        if self.valid_set is not None:
            self.modules.eval()
            epoch = self.hparams.epoch_counter.current
            step = self.step_counter.current
            avg_valid_loss = 0.0
            self.on_stage_start(sb.Stage.VALID, epoch)
            with torch.no_grad():
                for batch in tqdm(
                    self.valid_set, disable=True, leave=False
                ):
                    loss = self.evaluate_batch(batch, stage=sb.Stage.VALID)
                    avg_valid_loss += float(loss)

                avg_valid_loss /= len(self.valid_set)
                # Only run validation "on_stage_end" on main process
                # logger.info(f"Intra valid loss: {avg_valid_loss:.3f}")
                stage_stats = {
                    "loss": avg_valid_loss, 
                    "CER": self.cer_metric.summarize("error_rate"),
                    "WER": self.wer_metric.summarize("error_rate"),
                    "wer": self.hug_wer_metric.compute(),
                }

                self.hparams.train_logger.run.log(
                    {"valid": stage_stats},
                    step=step
                )
                self.hparams.train_logger_txt.log_stats(
                    stats_meta={
                        "epoch": epoch,
                        "step": step,
                    },
                    valid_stats=stage_stats,
                )
            
            self.modules.train()


# Define custom data procedure
def dataio_prepare(hparams, tokenizer):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""

    # 1. Define datasets
    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_csv"],
    )
    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_csv"],
    )
    test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["test_csv"],
    )

    # train_data, valid_data, test_data = preprocess_dataset(hparams, tokenizer)

    if hparams["sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        train_data = train_data.filtered_sorted(
            sort_key=hparams["sort_key"],
            key_min_value={hparams["sort_key"]: hparams["avoid_if_shorter_than"]},
            key_max_value={hparams["sort_key"]: hparams["avoid_if_longer_than"]},
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["dataloader_options"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key=hparams["sort_key"],
            reverse=True,
            key_min_value={hparams["sort_key"]: hparams["avoid_if_shorter_than"]},
            key_max_value={hparams["sort_key"]: hparams["avoid_if_longer_than"]},
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["dataloader_options"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        pass

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )

    # We also sort the validation data so it is faster to validate
    valid_data = valid_data.filtered_sorted(sort_key=hparams["sort_key"])

    # We also sort the validation data so it is faster to validate
    test_data = test_data.filtered_sorted(sort_key=hparams["sort_key"])

    datasets = [train_data, valid_data, test_data]

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        info = torchaudio.info(wav)
        sig = sb.dataio.dataio.read_audio(wav)

        if info.sample_rate == hparams["sample_rate"]:
            return sig

        resampled = torchaudio.transforms.Resample(
            info.sample_rate, hparams["sample_rate"],
        )(sig)
        return resampled

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("words")
    @sb.utils.data_pipeline.provides(
        "tokens_list", "tokens"
    )
    def text_pipeline(words):
        # tokens_list = words.split()
        # tokens_list = tokenizer.encode_as_ids(words)
        tokens_list = tokenizer._tokenize(words)
        tokens_list = [tokenizer.encoder.get(token) for token in tokens_list]
        
        yield tokens_list
        tokens = torch.IntTensor(tokens_list)
        yield tokens

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets, ["id", "sig", "tokens", "words", "wav", "duration"],
    )

    return train_data, valid_data, test_data

if __name__ == "__main__":

    # Load hyperparameters file with command-line overrides
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    print(run_opts)
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # If distributed_launch=True then
    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    # run_on_main(hparams["pretrainer"].collect_files)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Create the datasets objects as well as tokenization and encoding :-D
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(
        hparams['pretrained_tokenizer_path'])
    train_data, valid_data, test_data = dataio_prepare(hparams, tokenizer)

    # Trainer initialization
    asr_brain = ASR(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )
    # Adding objects to trainer.
    asr_brain.tokenizer = tokenizer

    # Training
    # with torch.autograd.detect_anomaly():
    asr_brain.fit(
        asr_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["test_dataloader_options"],
    )

    # Test
    asr_brain.hparams.wer_file = hparams["output_folder"] + "/wer_test.txt"
    if hparams["use_lm"]:
        print("load language model for testing")
        asr_brain.hparams.lm = get_decoder_ngram_model(tokenizer, hparams['lm_path'], hparams['lm_vocab_path'])
    asr_brain.evaluate(
        test_data,
        max_key="epoch",
        test_loader_kwargs=hparams["test_dataloader_options"],
    )