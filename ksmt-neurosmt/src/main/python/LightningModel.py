from sklearn.metrics import classification_report

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

from torchmetrics.classification import BinaryAccuracy, BinaryConfusionMatrix, BinaryAUROC, BinaryPrecisionAtFixedRecall

from Model import Model


def unpack_batch(batch):
    node_labels, edges, depths, root_ptrs = batch.x, batch.edge_index, batch.depth, batch.ptr

    return node_labels, edges, depths, root_ptrs


class LightningModel(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.model = Model()

        self.val_outputs = []
        self.val_targets = []

        self.acc = BinaryAccuracy()
        self.confusion_matrix = BinaryConfusionMatrix()

        self.roc_auc = BinaryAUROC()
        self.precisions_at_recall = nn.ModuleList([
            BinaryPrecisionAtFixedRecall(rec) for rec in [0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99, 1.0]
        ])

    def forward(self, node_labels, edges, depths, root_ptrs):
        return self.model(node_labels, edges, depths, root_ptrs)

    def configure_optimizers(self):
        params = [p for p in self.model.parameters() if p is not None and p.requires_grad]
        optimizer = torch.optim.Adam(params, lr=1e-4)

        return optimizer

    def training_step(self, train_batch, batch_idx):
        out = self.model(*unpack_batch(train_batch))
        loss = F.binary_cross_entropy_with_logits(out, train_batch.y.to(torch.float))

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

    def shared_val_test_step(self, batch, batch_idx, target_name):
        out = self.model(*unpack_batch(batch))
        loss = F.binary_cross_entropy_with_logits(out, batch.y.to(torch.float))

        out = F.sigmoid(out)

        self.roc_auc.update(out, batch.y)
        for precision_at_recall in self.precisions_at_recall:
            precision_at_recall.update(out, batch.y)

        self.log(
            f"{target_name}/loss", loss.float(),
            prog_bar=True, logger=True,
            on_step=False, on_epoch=True,
            batch_size=batch.num_graphs
        )
        self.log(
            f"{target_name}/acc", self.acc(out, batch.y),
            prog_bar=True, logger=True,
            on_step=False, on_epoch=True,
            batch_size=batch.num_graphs
        )

        self.val_outputs.append(out)
        self.val_targets.append(batch.y)

        return loss

    def shared_val_test_epoch_end(self, target_name):
        print("\n", flush=True)

        all_outputs = torch.flatten(torch.cat(self.val_outputs))
        all_targets = torch.flatten(torch.cat(self.val_targets))

        self.val_outputs.clear()
        self.val_targets.clear()

        self.print_confusion_matrix_and_classification_report(all_outputs, all_targets)

        self.log(
            f"{target_name}/roc-auc", self.roc_auc.compute(),
            prog_bar=True, logger=True,
            on_step=False, on_epoch=True
        )
        self.roc_auc.reset()

        for precision_at_recall in self.precisions_at_recall:
            precision = precision_at_recall.compute()[0]
            recall = precision_at_recall.min_recall

            self.log(
                f"{target_name}/precision_at_{recall}", precision,
                prog_bar=False, logger=True,
                on_step=False, on_epoch=True
            )

            precision_at_recall.reset()

    def validation_step(self, val_batch, batch_idx):
        return self.shared_val_test_step(val_batch, batch_idx, "val")

    def print_confusion_matrix_and_classification_report(self, all_outputs, all_targets):
        conf_mat = self.confusion_matrix(all_outputs, all_targets).detach().cpu().numpy()

        print("        +-------+-----------+-----------+")
        print("       ", "|", "unsat", "|", str(conf_mat[0][0]).rjust(9, " "), "|", str(conf_mat[0][1]).rjust(9, " "), "|")
        print("targets", "|", "  sat", "|", str(conf_mat[1][0]).rjust(9, " "), "|", str(conf_mat[1][1]).rjust(9, " "), "|")
        print("        +-------+-----------+-----------+")
        print("       ", "|", "     ", "|", "  unsat  ", "|", "   sat   ", "|")
        print("        +-------+-----------+-----------+")
        print("                      preds", "\n", sep="")

        all_outputs = all_outputs.float().cpu().numpy()
        all_targets = all_targets.float().cpu().numpy()

        all_outputs = all_outputs > 0.5
        print(
            classification_report(all_targets, all_outputs, target_names=["unsat", "sat"], digits=3, zero_division=0.0),
            flush=True
        )

    def on_validation_epoch_end(self):
        self.shared_val_test_epoch_end("val")

    def test_step(self, test_batch, batch_idx):
        return self.shared_val_test_step(test_batch, batch_idx, "test")

    def on_test_epoch_end(self):
        self.shared_val_test_epoch_end("test")
