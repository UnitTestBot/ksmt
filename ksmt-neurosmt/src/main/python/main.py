#!/usr/bin/python3

import os; os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"; os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["GPU"]
from argparse import ArgumentParser

import torch

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from GraphDataloader import load_data_from_scratch

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

    return args


if __name__ == "__main__":
    seed_everything(24, workers=True)
    torch.set_float32_matmul_precision("medium")

    args = get_args()
    tr, va, te = load_data_from_scratch(args.ds)

    pl_model = LightningModel()
    trainer = Trainer(
        accelerator="auto",
        # precision="bf16-mixed",
        logger=TensorBoardLogger("../logs", name="neuro-smt"),
        callbacks=[ModelCheckpoint(
            filename="epoch_{epoch:03d}_roc-auc_{val/roc-auc:.3f}",
            monitor="val/roc-auc",
            verbose=True,
            save_last=True, save_top_k=3, mode="max",
            auto_insert_metric_name=False, save_on_train_epoch_end=False
        )],
        max_epochs=200,
        log_every_n_steps=1,
        enable_checkpointing=True,
        barebones=False,
        default_root_dir=".."
    )

    if args.ckpt is None:
        trainer.fit(pl_model, tr, va)
    else:
        trainer.fit(pl_model, tr, va, ckpt_path=args.ckpt)
