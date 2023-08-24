#!/usr/bin/python3

import os; os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"; os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["GPU"]
from argparse import ArgumentParser

import torch

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from GlobalConstants import MAX_EPOCHS
from GraphDataloader import get_dataloader
from LightningModel import LightningModel


def get_args():
    parser = ArgumentParser(description="main training script")
    parser.add_argument("--ds", required=True, nargs="+")
    parser.add_argument("--oenc", required=True)
    parser.add_argument("--ckpt", required=False)

    args = parser.parse_args()
    print("args:")
    for arg in vars(args):
        print(arg, "=", getattr(args, arg))

    print()

    return args


if __name__ == "__main__":
    # enable usage of nvidia's TensorFloat if available
    torch.set_float32_matmul_precision("medium")

    args = get_args()

    train_dl = get_dataloader(args.ds, "train", args.oenc)
    val_dl = get_dataloader(args.ds, "val", args.oenc)

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
        max_epochs=MAX_EPOCHS,
        log_every_n_steps=1,
        enable_checkpointing=True,
        barebones=False,
        default_root_dir=".."
    )

    trainer.fit(pl_model, train_dl, val_dl, ckpt_path=args.ckpt)
