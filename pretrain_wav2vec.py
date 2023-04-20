#!/usr/bin/env python3
import sys
import os
import torch
import logging
import random
import speechbrain as sb
import torchaudio
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.checkpoints import torch_save, torch_recovery, register_checkpoint_hooks, mark_as_saver, mark_as_loader
from typing import Any, Dict, List, Optional, Union
from transformers import get_scheduler
from tqdm.contrib import tqdm
import math

from step_counter import StepCounter

"""Recipe for pre-training a Wav2Vec2 Model

Authors
 * Binh Le 2021
"""

logger = logging.getLogger(__name__)


class ASR(sb.core.Brain):
    def __init__(self, max_gumbel_temp=1, min_gumbel_temp=0, gumbel_temp_decay=1.0, *args, **kwargs):
        super(ASR, self).__init__(*args, **kwargs)
        self.step_counter = StepCounter()
        self.gradient_accumulation = self.hparams.gradient_accumulation
        self.accumulation_size = self.hparams.accumulation_sec # * self.hparams.sample_rate
        self.max_gumbel_temp = max_gumbel_temp
        self.min_gumbel_temp = min_gumbel_temp
        self.gumbel_temp_decay = gumbel_temp_decay
        self.gumbel_temp = max_gumbel_temp

        self.num_batch = -1
        self.valid_set = None

    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""

        wavs, wav_lens = batch.sig

        if stage == sb.Stage.TRAIN:
            # Augmentation
            if hasattr(self.hparams, "augmentation"):
                wavs = self.hparams.augmentation(wavs)

        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)

        if self.hparams.normalize_wav:
            with torch.no_grad():
                wavs = torch.nn.functional.layer_norm(
                    wavs, wavs.shape[1:], eps=1e-7)

        outputs, mask_time_indices = self.modules.wav2vec2(wavs)

        return outputs, mask_time_indices

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss (CTC) given predictions and targets."""
        outputs, mask_time_indices = predictions
        loss = outputs.loss
        if stage != sb.Stage.TRAIN:
            # We compute the accuracy between embeddings with cosing sim.
            num_losses = mask_time_indices.sum()
            cosine_sim = torch.cosine_similarity(
                outputs.projected_states, outputs.projected_quantized_states, dim=-1
            )
            acc = cosine_sim[mask_time_indices].mean()
            self.acc_metric.append(float(acc))
            self.contrast_loss.append(float(outputs.contrastive_loss.detach() / num_losses))
            self.div_loss.append(float(outputs.diversity_loss.detach() / num_losses))
            return loss.detach() / num_losses

        return loss

    def fit_batch(self, batch):
        """Train the parameters given a single batch in input"""
        if self.auto_mix_prec:
            with torch.cuda.amp.autocast():
                predictions = self.compute_forward(batch, sb.Stage.TRAIN)
                loss = self.compute_objectives(predictions, batch, sb.Stage.TRAIN)
                loss = loss / self.hparams.gradient_accumulation
            
            outputs, mask_time_indices = predictions
            num_losses = mask_time_indices.sum()

            # normalize the loss by gradient_accumulation step
            self.scaler.scale(loss).backward()

            if self.step % self.gradient_accumulation == 0 or self.step == self.num_batch:
                self.scaler.unscale_(self.model_optimizer)
                if self.check_gradients(loss):
                    self.scaler.step(self.model_optimizer)

                self.scheduler.step()

                self.modules.zero_grad(set_to_none=True)

                self.scaler.update()

                self.step_counter.update()
                self.update_gumbel_temperature()

                percent_masked = num_losses / torch.ones_like(mask_time_indices).sum()
                log_stats = {
                    "step_loss": loss.detach() * self.hparams.gradient_accumulation / num_losses,
                    "contrast_loss": outputs.contrastive_loss.detach() / num_losses,
                    "div_loss": outputs.diversity_loss.detach() / num_losses,
                    "%_mask_idx": percent_masked,
                    "ppl": outputs.codevector_perplexity,
                    "lr": self.model_optimizer.param_groups[0]["lr"],
                    "temp": self.gumbel_temp,
                    "num_losses": num_losses,
                    "accumulate": self.gradient_accumulation,
                }
                self.hparams.train_logger.run.log(
                    log_stats,
                    step=self.step_counter.current
                )

                if self.accumulation_size:
                    batch_duration = batch.duration[-1] * self.hparams.batch_size
                    self.gradient_accumulation = math.ceil(self.accumulation_size / batch_duration)
                    if self.gradient_accumulation < self.hparams.min_accumulation:
                        self.gradient_accumulation = self.hparams.min_accumulation
        else:
            predictions = self.compute_forward(batch, sb.Stage.TRAIN)
            loss = self.compute_objectives(predictions, batch, sb.Stage.TRAIN)
            loss = loss / self.hparams.gradient_accumulation
            
            outputs, mask_time_indices = predictions
            # normalize the loss by gradient_accumulation step
            loss.backward()

            num_losses = mask_time_indices.sum()
            if self.step % self.gradient_accumulation == 0 or self.step == self.num_batch:
                if self.check_gradients(loss):
                    self.model_optimizer.step()

                # self.hparams.scheduler(self.model_optimizer)
                self.scheduler.step()

                self.modules.zero_grad(set_to_none=True)

                self.step_counter.update()
                self.update_gumbel_temperature()

                percent_masked = num_losses / torch.ones_like(mask_time_indices).sum()
                log_stats = {
                    "step_loss": loss.detach() * self.hparams.gradient_accumulation / num_losses,
                    "contrast_loss": outputs.contrastive_loss.detach() / num_losses,
                    "div_loss": outputs.diversity_loss.detach() / num_losses,
                    "%_mask_idx": percent_masked,
                    "ppl": outputs.codevector_perplexity,
                    "lr": self.model_optimizer.param_groups[0]["lr"],
                    "temp": self.gumbel_temp,
                    "num_losses": num_losses,
                    "accumulate": self.gradient_accumulation,
                }
                self.hparams.train_logger.run.log(
                    log_stats,
                    step=self.step_counter.current
                )

                batch_duration = batch.duration[-1] * self.hparams.batch_size
                self.gradient_accumulation = math.ceil(self.accumulation_size / batch_duration)
                if self.gradient_accumulation < self.hparams.min_accumulation:
                    self.gradient_accumulation = self.hparams.min_accumulation
                

        return loss.detach() * self.hparams.gradient_accumulation / num_losses

    def evaluate_batch(self, batch, stage):
        """Computations needed for validation/test batches"""
        with torch.no_grad():
            predictions = self.compute_forward(batch, stage=stage)
            loss = self.compute_objectives(predictions, batch, stage=stage)
        return loss.detach()

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        if stage != sb.Stage.TRAIN:
            self.acc_metric = []
            self.contrast_loss = []
            self.div_loss = []
        else:
            self.gradient_accumulation = self.hparams.gradient_accumulation

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of an epoch."""
        # Compute/store important stats
        step = self.step_counter.current
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["acc"] = sum(self.acc_metric) / len(self.acc_metric)
            stage_stats["contrast_loss"] = sum(self.contrast_loss) / len(self.contrast_loss)
            stage_stats["div_loss"] = sum(self.div_loss) / len(self.div_loss)

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            old_lr_model = self.scheduler.get_last_lr()[0]
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
                meta={"loss": stage_stats["loss"], "epoch": epoch}, min_keys=["loss"],
            )
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger_txt.log_stats(
                stats_meta={
                    "Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
    
    def _save_intra_epoch_ckpt(self):
        """Saves a CKPT with specific intra-epoch flag."""
        super()._save_intra_epoch_ckpt()
        # self._intra_evaluate()
    
    def _intra_evaluate(self):
        # Validation stage
        if self.valid_set is not None and random.random() < 0.5:
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
                    "acc": sum(self.acc_metric) / len(self.valid_set),
                    "contrast_loss": sum(self.contrast_loss) / len(self.valid_set),
                    "div_loss": sum(self.div_loss) / len(self.valid_set),
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
                    train_stats={},
                    valid_stats=stage_stats,
                )

            self.modules.train()

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
        self.num_batch = math.ceil(len(train_set) / batch_size) +8273
        super().fit(
            epoch_counter,
            train_set,
            valid_set,
            progressbar,
            train_loader_kwargs,
            valid_loader_kwargs,
        )

    def init_optimizers(self):
        "Initializes the wav2vec2 optimizer and model optimizer"
        num_training_steps = self.hparams.num_training_steps
        self.model_optimizer = self.hparams.model_opt_class(
            self.modules.parameters()
        )
        self.scheduler = get_scheduler(
            "linear",
            optimizer=self.model_optimizer,
            num_warmup_steps=0.08*num_training_steps,
            num_training_steps=num_training_steps
        )

        if self.checkpointer is not None:
            self.checkpointer.add_recoverable("modelopt", self.model_optimizer)
            self.checkpointer.add_recoverable(
                "scheduler",
                self.scheduler,
                custom_load_hook=torch_recovery,
                custom_save_hook=torch_save)
            self.checkpointer.add_recoverable("update_step", self.step_counter)
    
    def update_gumbel_temperature(self):
        self.gumbel_temp = self.max_gumbel_temp * self.gumbel_temp_decay ** self.step_counter.current
        if self.gumbel_temp >= self.min_gumbel_temp:
            self.modules.wav2vec2.model.set_gumbel_temperature(self.gumbel_temp)
        else:
            self.gumbel_temp = self.min_gumbel_temp


# Define custom data procedure
def dataio_prepare(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""

    # 1. Define datasets
    root_folder = hparams["root_folder"]
    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_csv"], replacements={"data_folder": root_folder}
    )
    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_csv"],
    )
    test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["test_csv"],
    )

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
    test_data = test_data.filtered_sorted(
        sort_key=hparams["sort_key"],
        key_min_value={hparams["sort_key"]: 1}
    )

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

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets, ["id", "sig", "duration"],
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


    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Create the datasets objects as well as tokenization and encoding :-D
    train_data, valid_data, test_data = dataio_prepare(hparams)

    modules = hparams["modules"]
    # modules['wav2vec2'].model.load_state_dict(torch.load(hparams['wav2vec2_hub'] + "/pytorch_model.bin", map_location="cpu"))

    # Trainer initialization
    asr_brain = ASR(
        modules=modules,
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
        max_gumbel_temp=hparams["max_gumbel_temperature"],
        min_gumbel_temp=hparams["min_gumbel_temperature"],
        gumbel_temp_decay=hparams["gumbel_temperature_decay"],
    )

    # Training
    asr_brain.fit(
        asr_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["test_dataloader_options"],
    )

    # Test
    asr_brain.evaluate(
        test_data,
        max_key="epoch",
        test_loader_kwargs=hparams["test_dataloader_options"],
    )
