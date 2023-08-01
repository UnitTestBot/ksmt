from sklearn.metrics import classification_report

import torch
import torch.nn.functional as F

import pytorch_lightning as pl

from torchmetrics.classification import BinaryAccuracy, BinaryAUROC

from Model import Model


class LightningModel(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.model = Model()

        self.val_outputs = []
        self.val_targets = []

        self.acc = BinaryAccuracy()
        self.roc_auc = BinaryAUROC()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        params = [p for p in self.model.parameters() if p is not None and p.requires_grad]
        optimizer = torch.optim.Adam(params, lr=1e-4)

        return optimizer

    def training_step(self, train_batch, batch_idx):
        out = self.model(train_batch)
        loss = F.binary_cross_entropy_with_logits(out, train_batch.y)

        out = F.sigmoid(out)

        self.log(
            "train/loss", loss.detach().float(),
            prog_bar=True, logger=True,
            on_step=True, on_epoch=True,
            batch_size=train_batch.num_graphs
        )
        self.log(
            "train/acc", self.acc(out, train_batch.y),
            prog_bar=True, logger=True,
            on_step=False, on_epoch=True,
            batch_size=train_batch.num_graphs
        )

        return loss

    def shared_val_test_step(self, batch, batch_idx, metric_name):
        out = self.model(batch)
        loss = F.binary_cross_entropy_with_logits(out, batch.y)

        out = F.sigmoid(out)

        self.log(
            f"{metric_name}/loss", loss.float(),
            prog_bar=True, logger=True,
            on_step=False, on_epoch=True,
            batch_size=batch.num_graphs
        )
        self.log(
            f"{metric_name}/acc", self.acc(out, batch.y),
            prog_bar=True, logger=True,
            on_step=False, on_epoch=True,
            batch_size=batch.num_graphs
        )

        self.val_outputs.append(out)
        self.val_targets.append(batch.y)

        return loss

    def validation_step(self, val_batch, batch_idx):
        return self.shared_val_test_step(val_batch, batch_idx, "val")

    def on_validation_epoch_end(self):
        print("\n\n", flush=True)

        all_outputs = torch.flatten(torch.cat(self.val_outputs))
        all_targets = torch.flatten(torch.cat(self.val_targets))

        self.val_outputs.clear()
        self.val_targets.clear()

        self.log(
            "val/roc-auc", self.roc_auc(all_outputs, all_targets),
            prog_bar=True, logger=True,
            on_step=False, on_epoch=True
        )

        all_outputs = all_outputs.float().cpu().numpy()
        all_targets = all_targets.float().cpu().numpy()

        all_outputs = all_outputs > 0.5
        print(classification_report(all_targets, all_outputs, digits=3, zero_division=0.0))

        print("\n", flush=True)

    def test_step(self, test_batch, batch_idx):
        return self.shared_val_test_step(test_batch, batch_idx, "test")

    def on_test_epoch_end(self):
        all_outputs = torch.flatten(torch.cat(self.val_outputs))
        all_targets = torch.flatten(torch.cat(self.val_targets))

        self.val_outputs.clear()
        self.val_targets.clear()

        self.log(
            "test/roc-auc", self.roc_auc(all_outputs, all_targets),
            prog_bar=True, logger=True,
            on_step=False, on_epoch=True
        )
