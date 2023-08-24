#!/usr/bin/python3

import sys

import torch

from GlobalConstants import EMBEDDING_DIM
from LightningModel import LightningModel


if __name__ == "__main__":

    pl_model = LightningModel().load_from_checkpoint(sys.argv[1], map_location=torch.device("cpu"))
    pl_model.eval()

    # input example
    node_labels = torch.unsqueeze(torch.arange(0, 9).to(torch.int32), 1)
    node_features = pl_model.model.encoder.embedding(node_labels.squeeze())
    edges = torch.tensor([
        [0, 0, 0, 1, 2, 3, 4, 4, 5, 6, 7],
        [1, 2, 3, 3, 7, 4, 5, 6, 8, 8, 8]
    ], dtype=torch.int64)

    torch.onnx.export(
        pl_model.model.encoder.embedding,
        (node_labels,),
        sys.argv[2],  # path to model file
        opset_version=18,
        input_names=["node_labels"],
        output_names=["out"],
        dynamic_axes={
            "node_labels": {0: "nodes_number"}
        }
    )

    torch.onnx.export(
        pl_model.model.encoder.conv,
        (node_features, edges),
        sys.argv[3],  # path to model file
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
        sys.argv[4],  # path to model file
        opset_version=18,
        input_names=["expr_features"],
        output_names=["out"],
        dynamic_axes={
            "expr_features": {0: "batch_size"}
        }
    )

    print("success!")
