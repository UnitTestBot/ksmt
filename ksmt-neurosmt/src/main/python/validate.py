#!/usr/bin/python3

import os; os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"; os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["GPU"]
from argparse import ArgumentParser

import torch

from pytorch_lightning import Trainer, seed_everything

from GraphDataloader import load_data_from_scratch

from LightningModel import LightningModel


def get_args():
    parser = ArgumentParser(description="validation script")
    parser.add_argument("--ds", required=True)
    parser.add_argument("--ckpt", required=True)

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

    trainer = Trainer()

    trainer.validate(LightningModel(), va, args.ckpt)
    trainer.test(LightningModel(), te, args.ckpt)
