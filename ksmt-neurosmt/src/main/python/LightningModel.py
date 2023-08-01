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

        #self.train_dl = train_dl
        #self.val_dl = val_dl
        #self.test_dl = test_dl

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

    def validation_step(self, val_batch, batch_idx):
        out = self.model(val_batch)
        loss = F.binary_cross_entropy_with_logits(out, val_batch.y)

        out = F.sigmoid(out)

        self.log(
            "val/loss", loss.float(),
            prog_bar=True, logger=True,
            on_step=False, on_epoch=True,
            batch_size=val_batch.num_graphs
        )
        self.log(
            "val/acc", self.acc(out, val_batch.y),
            prog_bar=True, logger=True,
            on_step=False, on_epoch=True,
            batch_size=val_batch.num_graphs
        )

        self.val_outputs.append(out)
        self.val_targets.append(val_batch.y)

        """
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
        """

        return loss

    def on_validation_epoch_end(self):
        print("\n\n", flush=True)

        all_outputs = torch.flatten(torch.cat(self.val_outputs))
        all_targets = torch.flatten(torch.cat(self.val_targets))

        self.val_outputs.clear()
        self.val_targets.clear()

        logger = self.logger.experiment

        roc_auc = self.roc_auc(all_outputs, all_targets)
        self.log(
            "val/roc-auc", roc_auc,
            prog_bar=True, logger=False,
            on_step=False, on_epoch=True
        )
        logger.add_scalar("val/roc-auc", roc_auc, self.current_epoch)

        all_outputs = all_outputs.float().cpu().numpy()
        all_targets = all_targets.float().cpu().numpy()

        all_outputs = all_outputs > 0.5
        print(classification_report(all_targets, all_outputs, digits=3, zero_division=0.0))

        print("\n", flush=True)

    """
    def backward(self, trainer, loss, optimizer, optimizer_idx):
        loss.backward()
    """

    """
    def train_dataloader(self):
        return self.train_dl

    def val_dataloader(self):
        return self.val_dl

    def test_dataloader(self):
        return self.test_dl
    """
