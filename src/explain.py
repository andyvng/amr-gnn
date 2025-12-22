import glob
import os

import hydra
import lightning as L
import torch_geometric.loader as geom_loader
from torch_geometric.explain import Explainer, GNNExplainer, ModelConfig

from models import AMRNodeDualGNN
from utils import create_graph_dataset


@hydra.main(version_base=None, config_path="../conf", config_name="explain")
def main(cfg):
    L.seed_everything(cfg.trainer.seed * cfg.data.run_id)

    # Create outdir directory
    os.makedirs(cfg.prediction.outdir, exist_ok=True)

    # Dataset & Dataloader
    dataset, adj_mat_1, adj_mat_2 = create_graph_dataset(cfg)
    c_in = dataset.x.size(-1)
    node_data_loader = geom_loader.DataLoader([dataset], batch_size=1)

    if cfg.data.use_class_weight:
        class_weight = dataset.y.eq(0).sum(dim=0) / dataset.y.eq(1).sum(dim=0)
    else:
        class_weight = None

    # Load model from checkpoint
    checkpoint = glob.glob(os.path.join(cfg.trainer.model_checkpoint.dirpath, "*.ckpt"))
    print(checkpoint[0])
    assert len(checkpoint) == 1, (
        "There should be exactly one checkpoint file in the specified directory."
    )
    model = AMRNodeDualGNN.load_from_checkpoint(
        checkpoint[0], cfg=cfg, c_in=c_in, class_weight=class_weight
    )

    # Create trainer
    trainer = L.Trainer(
        accelerator="auto", devices="auto", logger=False, enable_checkpointing=False
    )

    print("Predict AMR phenotype...")
    # Prediction
    trainer.predict(model, dataloaders=node_data_loader)


if __name__ == "__main__":
    main()


# @hydra.main(version_base=None, config_path="../conf/dual_gnn", config_name="explain")
# def main(cfg):
#     # find checkpoint
#     model_ckpts = glob.glob(
#         f"{cfg.trainer.model_checkpoint.dirpath}/**/*.ckpt", recursive=True
#     )
#     assert len(model_ckpts) == 1, "Only 1 best model checkpoint should be found"
#     CKPT_PATH = model_ckpts[0]

#     dataset, _, _ = create_graph_dataset(cfg)
#     # TODO
#     node_id_df = pd.DataFrame({"node_id": dataset.ids, "id": dataset.isolate_codes})
#     c_in = dataset.x.size(-1)
#     model_kwargs = {"heads": cfg.gnn.GAT.heads}  # If GAT is used
#     model = GNNModel(cfg=cfg, c_in=c_in, **model_kwargs)

#     if torch.cuda.is_available():
#         device = torch.device("cuda")
#     elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
#         print("Using MPS")
#         device = torch.device("mps")
#     else:
#         device = torch.device("cpu")

#     checkpoint = torch.load(CKPT_PATH, map_location=device)
#     model_weights = {
#         k.lstrip("model"): v
#         for k, v in checkpoint["state_dict"].items()
#         if k.startswith("model.")
#     }
#     model_weights = {k.lstrip("."): v for k, v in model_weights.items()}
#     model.load_state_dict(model_weights)
#     model.eval()

#     # DeepLiftSHAP
#     # According to captum, no need to specify target for binary classification
#     # According to the DeepLift paper, prefer logit layer to softmax/sigmoid layer
#     # Need a reference input (PAO1)

#     baseline = torch.load(cfg.explainer.baseline).to(device)

#     if cfg.explainer.algorithm == "DeepLIFTSHAP":
#         dl = DeepLiftShap(model)
#     elif cfg.explainer.algorithm == "DeepLIFT":
#         dl = DeepLift(model)
#     else:
#         dl = IntegratedGradients(model)

#     x, edge_index_1, edge_index_2 = (
#         dataset.x.to(device),
#         dataset.edge_index_1.to(device),
#         dataset.edge_index_2.to(device),
#     )

#     attribution = dl.attribute(
#         x,
#         target=None,
#         baselines=baseline,
#         additional_forward_args=(edge_index_1, edge_index_2),
#     )

#     # Save attribution with node ids and isolate codes
#     tmp_attribution_df = pd.DataFrame(attribution.detach().cpu().numpy())
#     attribution_df = pd.concat([node_id_df, tmp_attribution_df], axis=1)
#     attribution_df.to_csv(cfg.explainer.out_fp, index=False)
