#!/usr/bin/python3

import os; os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"; os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["GPU"]
import time
from argparse import ArgumentParser

import torch

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger

from GraphDataloader import load_data

from LightningModel import LightningModel


def get_args():
    parser = ArgumentParser(description="main training script")
    parser.add_argument("--ds", required=True)
    parser.add_argument("--ckpt", required=False)

    args = parser.parse_args()
    print("args:")
    for arg in vars(args):
        print(arg, "=", getattr(args, arg))

    print()
    time.sleep(5)

    return args


if __name__ == "__main__":
    seed_everything(24, workers=True)
    torch.set_float32_matmul_precision("medium")

    args = get_args()
    tr, va, te = load_data(args.ds)

    pl_model = LightningModel()
    trainer = Trainer(
        accelerator="auto",
        # precision="bf16-mixed",
        logger=TensorBoardLogger("../logs", name="neuro-smt"),
        max_epochs=100,
        log_every_n_steps=1,
        enable_checkpointing=True,
        barebones=False,
        default_root_dir=".."
    )

    if args.ckpt is None:
        trainer.fit(pl_model, tr, va)
    else:
        trainer.fit(pl_model, tr, va, ckpt_path=args.ckpt)
