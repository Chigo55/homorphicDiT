import os
import random
import torch

from pathlib import Path

from engine import LightningEngine
from model.model import *

# main.py 맨 위
import torch
torch.autograd.set_detect_anomaly(mode=True)


def get_hparams():
    hparams = {
        # 모델 구조
        "image_size": 512,
        "offset": 0.5,
        "cutoff": 0.2,

        # 손실 함수 가중치 (losses.py 기준)
        "lambda_col": 10.0,
        "lambda_exp": 1.0,
        "lambda_spa": 100.0,

        # 최적화 및 학습 설정
        "lr": 1e-6,
        "decay": 1e-7,
        "epochs": 100,
        "patience": 30,
        "batch_size": 16,
        "seed": random.randint(a=0, b=1000),

        # 데이터 경로
        "train_data_path": "data/1_train",
        "valid_data_path": "data/2_valid",
        "bench_data_path": "data/3_bench",
        "infer_data_path": "data/4_infer",

        # 로깅 설정
        "log_dir": "./runs/HomomorphicUnet",
        "experiment_name": "Base",
        "inference": "inference",
    }
    return hparams


def main():
    hparams = get_hparams()
    model = HomomorphicUnetLightning
    engin = LightningEngine(
        model=model,
        hparams=hparams,
        # ckpt="runs/HomomorphicUnet/Base/version_23/checkpoints/step-step=1000.ckpt"
    )

    print("[RUNNING] Trainer...")
    engin.train()

    print("[RUNNING] Validater...")
    engin.valid()

    print("[RUNNING] Benchmarker...")
    engin.bench()

    print("[RUNNING] Inferencer...")
    engin.infer()


if __name__ == "__main__":
    main()
