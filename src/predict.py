import glob
import os

import hydra
import lightning as L
import torch_geometric.loader as geom_loader

from models import AMRNodeDualGNN
from utils import create_graph_dataset


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg):
    L.seed_everything(cfg.trainer.seed * cfg.data.run_id)

    # Ensure predict IDs file exists
    assert os.path.exists(cfg.data.predict_ids), "Predict IDs file does not exist."

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
