import glob
import os

import hydra
import lightning as L
import torch_geometric.loader as geom_loader
from lightning.pytorch.callbacks import ModelCheckpoint

from models import AMRNodeDualGNN
from utils import create_graph_dataset


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg):
    L.seed_everything(cfg.trainer.seed)

    # Dataset & Dataloader
    dataset, adj_mat_1, adj_mat_2 = create_graph_dataset(cfg)
    c_in = dataset.x.size(-1)
    node_data_loader = geom_loader.DataLoader([dataset], batch_size=1)

    if cfg.data.use_class_weight:
        class_weight = dataset.y.eq(0).sum(dim=0) / dataset.y.eq(1).sum(dim=0)
    else:
        class_weight = None

    # Model
    model = AMRNodeDualGNN(cfg=cfg, c_in=c_in, class_weight=class_weight)

    # Create trainer
    os.makedirs(cfg.trainer.model_checkpoint.dirpath, exist_ok=True)

    model_ckpt = ModelCheckpoint(
        dirpath=cfg.trainer.model_checkpoint.dirpath,
        mode=cfg.trainer.model_checkpoint.mode,
        monitor=cfg.trainer.model_checkpoint.monitor,
        save_last=cfg.trainer.model_checkpoint.save_last,
        save_top_k=cfg.trainer.model_checkpoint.save_top_k,
    )

    logger = False

    trainer = L.Trainer(
        default_root_dir=cfg.trainer.default_root_dir,
        max_epochs=cfg.trainer.max_epochs,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        num_nodes=cfg.trainer.num_nodes,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        strategy=cfg.trainer.strategy,
        precision=cfg.trainer.precision,
        callbacks=[model_ckpt],
        logger=logger,
    )

    # Training and save
    trainer.fit(
        model, train_dataloaders=node_data_loader, val_dataloaders=node_data_loader
    )

    # Load the best checkpoint and evaluate on test set if provided
    if os.path.exists(cfg.data.test_ids):
        trainer.test(ckpt_path="best", dataloaders=node_data_loader)

    # Clean up checkpoint files to save storage
    if cfg.trainer.del_ckpt:
        ckpt_files = glob.glob(
            os.path.join(cfg.trainer.model_checkpoint.dirpath, "*.ckpt")
        )
        for file in ckpt_files:
            os.remove(file)


if __name__ == "__main__":
    print("Training started... with Hydra config.")
    main()
