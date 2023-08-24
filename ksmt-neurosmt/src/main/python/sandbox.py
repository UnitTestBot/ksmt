#!/usr/bin/python3

import sys
# import os; os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"; os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["GPU"]

import torch

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from GraphDataloader import get_dataloader

from LightningModel import LightningModel

from GlobalConstants import EMBEDDING_DIM


if __name__ == "__main__":

    pl_model = LightningModel().load_from_checkpoint(sys.argv[1], map_location=torch.device("cpu"))
    pl_model.eval()

    """
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
    """

    MAX_SIZE = 3

    node_labels = torch.tensor([[i] for i in range(MAX_SIZE)], dtype=torch.int32)
    edges = torch.tensor([
        [i for i in range(MAX_SIZE - 1)],
        [i for i in range(1, MAX_SIZE)]
    ], dtype=torch.int64)
    depths = torch.tensor([MAX_SIZE - 1], dtype=torch.int64)
    root_ptrs = torch.tensor([0, MAX_SIZE], dtype=torch.int64)

    torch.onnx.export(
        pl_model.model.encoder.embedding,
        (node_labels,),
        "embeddings.onnx",
        opset_version=18,
        input_names=["node_labels"],
        output_names=["out"],
        dynamic_axes={
            "node_labels": {0: "nodes_number"}
        }
    )

    torch.onnx.export(
        pl_model.model.encoder.conv,
        (torch.rand((node_labels.shape[0], EMBEDDING_DIM)), edges),
        "conv.onnx",
        opset_version=18,
        input_names=["node_features", "edges"],
        output_names=["out"],
        dynamic_axes={
            "node_features": {0: "nodes_number"},
            "edges": {1: "edges_number"}
        }
    )

    torch.onnx.export(
        pl_model.model.decoder,
        (torch.rand((1, EMBEDDING_DIM)),),
        "decoder.onnx",
        opset_version=18,
        input_names=["expr_features"],
        output_names=["out"],
        dynamic_axes={
            "expr_features": {0: "batch_size"}
        }
    )

    """
    pl_model.to_onnx(
        "kek.onnx",
        (node_labels, edges, depths, root_ptrs),
        opset_version=18,
        input_names=["node_labels", "edges", "depths", "root_ptrs"],
        output_names=["output"],
        dynamic_axes={
            "node_labels": {0: "nodes_number"},
            "edges": {1: "edges_number"},
            "depths": {0: "batch_size"},
            "root_ptrs": {0: "batch_size_+_1"},
        },
        # verbose=True
    )
    """
