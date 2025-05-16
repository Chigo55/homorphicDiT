import lightning as L
import torch
import torch.nn as nn


from lightning import Trainer
from lightning.pytorch.callbacks import *
from lightning.pytorch.loggers import TensorBoardLogger
from pathlib import Path

from engine.trainer import LightningTrainer
from engine.validater import LightningValidater
from engine.benchmarker import LightningBenchmarker
from engine.inferencer import LightningInferencer


class DetectNanCallback(L.Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        for name, param in pl_module.named_parameters():
            if torch.isnan(input=param.data).any():
                trainer.logger.experiment.add_text(
                    "NaN", f"param {name}", batch_idx)
                raise RuntimeError(f"NaN in param {name}")
            if param.grad is not None and torch.isnan(input=param.grad).any():
                raise RuntimeError(f"NaN in grad {name}")


class LightningEngine:
    def __init__(self, model: nn.Module, hparams: dict, ckpt: str = None):
        self.model = model
        self.hparams = hparams

        if ckpt:
            self.ckpt = Path(ckpt)
        else:
            self.ckpt = ckpt

        # --- 로깅 설정
        self.logger = self._build_logger()

        # --- 콜백 정의
        self.callbacks = self._build_callbacks()

        # --- Lightning Trainer 정의
        self.trainer = Trainer(
            max_epochs=hparams["epochs"],
            accelerator="gpu",
            devices=1,
            precision="32",
            logger=self.logger,
            callbacks=self.callbacks,
            log_every_n_steps=5,
        )

    def _build_logger(self):
        return TensorBoardLogger(
            save_dir=self.hparams["log_dir"],
            name=self.hparams["experiment_name"]
        )

    def _build_callbacks(self):
        return [
            ModelCheckpoint(
                monitor="valid/5_tot",
                save_top_k=1,
                mode="min",
                filename="best-{epoch:02d}",
            ),
            ModelCheckpoint(
                every_n_epochs=1,
                save_top_k=-1,
                filename="epoch-{epoch:02d}",
            ),
            ModelCheckpoint(
                every_n_train_steps=25,
                save_top_k=-1,
                filename="step-{step:04d}",
            ),
            EarlyStopping(
                monitor="valid/5_tot",
                patience=self.hparams["patience"],
                mode="min",
                verbose=True,
            ),
            LearningRateMonitor(logging_interval="step"),
            # DetectNanCallback(),
        ]

    def train(self):
        LightningTrainer(
            model=self.model,
            trainer=self.trainer,
            hparams=self.hparams,
            ckpt=self.ckpt
        ).run()

    def valid(self):
        LightningValidater(
            model=self.model,
            trainer=self.trainer,
            hparams=self.hparams,
            ckpt=self.ckpt
        ).run()

    def bench(self):
        LightningBenchmarker(
            model=self.model,
            trainer=self.trainer,
            hparams=self.hparams,
            ckpt=self.ckpt
        ).run()

    def infer(self):
        LightningInferencer(
            model=self.model,
            trainer=self.trainer,
            hparams=self.hparams,
            ckpt=self.ckpt
        ).run()
