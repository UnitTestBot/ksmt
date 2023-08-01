#!/usr/bin/python3

import sys
import os; os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"; os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["GPU"]
import time

import numpy as np
from tqdm import tqdm, trange

from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger

from GraphDataloader import load_data

from Model import Model
from LightningModel import LightningModel


if __name__ == "__main__":
    seed_everything(24, workers=True)
    torch.set_float32_matmul_precision("medium")

    tr, va, te = load_data(sys.argv[1])

    pl_model = LightningModel()
    trainer = Trainer(
        accelerator="auto",
        # precision="bf16-mixed",
        logger=TensorBoardLogger("../logs", name="neuro-smt"),
        max_epochs=100,
        log_every_n_steps=1,
        enable_checkpointing=False,
        barebones=False
    )

    trainer.fit(pl_model, tr, va)

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model().to(device)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p is not None and p.requires_grad], lr=1e-4)

    def calc_grad_norm():
        grads = [
            p.grad.detach().flatten() for p in model.parameters() if p.grad is not None and p.requires_grad
        ]
        return torch.cat(grads).norm().item()

    for p in model.parameters():
        assert p.requires_grad

    with open("log.txt", "a") as f:
        f.write("\n" + "=" * 12 + "\n")

    for epoch in trange(200):
        model.train()
        for batch in tqdm(tr):
            optimizer.zero_grad()
            batch = batch.to(device)

            out = model(batch)
            #out = out[batch.ptr[1:] - 1]

            loss = F.binary_cross_entropy_with_logits(out, batch.y)
            loss.backward()

            optimizer.step()

        print("\n", flush=True)
        print(f"grad norm: {calc_grad_norm()}")

        def validate(dl, val=False):
            model.eval()

            probas = torch.tensor([]).to(device)
            answers = torch.tensor([]).to(device)
            targets = torch.tensor([]).to(device)
            losses = []

            with torch.no_grad():
                for batch in tqdm(dl):
                    batch = batch.to(device)

                    out = model(batch)
                    #out = out[batch.ptr[1:] - 1]

                    loss = F.binary_cross_entropy_with_logits(out, batch.y)

                    out = F.sigmoid(out)
                    probas = torch.cat((probas, out))
                    out = (out > 0.5)

                    answers = torch.cat((answers, out))
                    targets = torch.cat((targets, batch.y.to(torch.int).to(torch.bool)))
                    losses.append(loss.item())

            probas = torch.flatten(probas).cpu().numpy()
            answers = torch.flatten(answers).cpu().numpy()
            targets = torch.flatten(targets).cpu().numpy()

            mean_loss = np.mean(losses)
            roc_auc = roc_auc_score(targets, probas) if val else None

            print("\n", flush=True)
            print(f"mean loss: {mean_loss}")
            print(f"acc: {accuracy_score(targets, answers)}")
            print(f"roc-auc: {roc_auc}")
            print(classification_report(targets, answers, digits=3, zero_division=0.0), flush=True)

            if val:
                return mean_loss, roc_auc
            else:
                return mean_loss

        print()
        print("train:")
        tr_loss = validate(tr)
        print("val:")
        va_loss, roc_auc = validate(va, val=True)
        print()

        with open("log.txt", "a") as f:
            tr_loss = "{:.9f}".format(tr_loss)
            va_loss = "{:.9f}".format(va_loss)
            roc_auc = "{:.9f}".format(roc_auc)

            f.write(f"{str(epoch).rjust(3)}: {tr_loss} | {va_loss} | {roc_auc}\n")
    
    """