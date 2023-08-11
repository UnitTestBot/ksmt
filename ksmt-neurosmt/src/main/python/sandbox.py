#!/usr/bin/python3

import sys
# import os; os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"; os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["GPU"]

import torch

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from GraphDataloader import get_dataloader

from LightningModel import LightningModel

from Model import EMBEDDING_DIM


if __name__ == "__main__":

    pl_model = LightningModel().load_from_checkpoint(sys.argv[1], map_location=torch.device("cpu"))

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

    print(node_labels.shape, edges.shape, depths.shape, root_ptrs.shape)

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

    """
    torch.onnx.export(torch_model,
                      x,
                      '../Game_env/gnn_model.onnx',
                      opset_version=15,
                      export_params=True,
                      input_names = ['x', 'edge_index'],   # the model's input names
                      output_names = ['output'],
                      dynamic_axes={'x' : {0 : 'nodes_number'},    # variable length axes
                                    'edge_index' : {1 : 'egdes_number'},
                                    },
                      )
    """

    """
    x = torch.randn(*shape, requires_grad=True).to(device)
    torch_model = actor_model.eval()
    torch_out = torch_model(x)
    torch.onnx.export(torch_model,
                      x,
                      '../Game_env/actor_model.onnx',
                      opset_version=15,
                      export_params=True,
                      input_names = ['input'],   # the model's input names
                      output_names = ['output'],
                      dynamic_axes={'input' : {0 : 'batch_size',
                                               1 : 'n_actions',
                                               },    # variable length axes
                                    'output' : {0 : 'batch_size',
                                                1 : 'n_actions',
                                                },
                                    },
                      )
    algo_name = 'NN'

    if actor_model is not None and gnn_model is not None and self.use_gnn:
        x_shape = [1, self.gnn_in_nfeatures]
        edge_index_shape = [2, 1]
        x = (torch.randn(*x_shape, requires_grad=True).to(device), torch.randint(0, 1, edge_index_shape).to(device))
        torch_model = gnn_model.eval()
        torch_out = torch_model(*x)
        torch.onnx.export(torch_model,
                          x,
                          '../Game_env/gnn_model.onnx',
                          opset_version=15,
                          export_params=True,
                          input_names = ['x', 'edge_index'],   # the model's input names
                          output_names = ['output'],
                          dynamic_axes={'x' : {0 : 'nodes_number'},    # variable length axes
                                        'edge_index' : {1 : 'egdes_number'},
                                        },
                          )
    """
