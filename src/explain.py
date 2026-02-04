import glob
import os

import hydra
import lightning as L
import pandas as pd
import torch
from captum.attr import IntegratedGradients

from models import GNNModel
from utils import create_graph_dataset


@hydra.main(version_base=None, config_path="../conf", config_name="explain")
def main(cfg):
    # Dataset & Dataloader
    dataset, adj_mat_1, adj_mat_2 = create_graph_dataset(cfg)
    node_id_df = pd.DataFrame({"id": dataset.isolate_codes})
    c_in = dataset.x.size(-1)

    # Load model from checkpoint
    model_kwargs = {}

    if cfg.gnn.layer_name in ["GAT", "GATv2"]:
        model_kwargs["heads"] = cfg.gnn.GAT.heads
    elif cfg.gnn.layer_name == "TransformerConv":
        model_kwargs["heads"] = cfg.gnn.TransformerConv.heads
    model = GNNModel(cfg=cfg, c_in=c_in, **model_kwargs)

    checkpoint = glob.glob(os.path.join(cfg.trainer.model_checkpoint.dirpath, "*.ckpt"))
    assert len(checkpoint) == 1, (
        "There should be exactly one checkpoint file in the specified directory."
    )
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("Using MPS")
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    checkpoint = torch.load(checkpoint[0], map_location=device)
    model_weights = {
        k.lstrip("model"): v
        for k, v in checkpoint["state_dict"].items()
        if k.startswith("model.")
    }
    model_weights = {k.lstrip("."): v for k, v in model_weights.items()}
    model.load_state_dict(model_weights)
    model.eval()
    model.to(device)

    if cfg.explainer.baseline.endswith(".pt"):
        baseline = torch.load(cfg.explainer.baseline).to(device)
    else:
        baseline = None  # Captum will use zero baseline by default

    dl = IntegratedGradients(model)

    x, edge_index_1, edge_index_2 = (
        dataset.x.to(device),
        dataset.edge_index_1.to(device),
        dataset.edge_index_2.to(device),
    )

    attribution = dl.attribute(
        x,
        target=None,
        baselines=baseline,
        additional_forward_args=(edge_index_1, edge_index_2),
        internal_batch_size=x.size(0),
        n_steps=cfg.explainer.n_steps,
    )

    # Save attribution with node ids and isolate codes
    tmp_attribution_df = pd.DataFrame(attribution.detach().cpu().numpy())
    attribution_df: pd.DataFrame = pd.concat([node_id_df, tmp_attribution_df], axis=1)
    attribution_df.to_csv(cfg.explainer.out_fp, index=False)


if __name__ == "__main__":
    main()
