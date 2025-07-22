import os
import sys
import glob
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from torch import nn, einsum
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from einops import rearrange, repeat
from einops.layers.torch import Reduce
from torchmetrics import AUROC
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
import hydra
from omegaconf import DictConfig, OmegaConf

import torch_geometric
import torch_geometric.nn as geom_nn
import torch_geometric.data as geom_data
import torch_geometric.loader as geom_loader

gnn_layer_by_name = {
    "GCN": geom_nn.GCNConv,
    "GAT": geom_nn.GATConv,
    "GATv2": geom_nn.GATv2Conv,
    "GraphConv": geom_nn.GraphConv,
    "TransformerConv": geom_nn.TransformerConv,
    "SAGE": geom_nn.SAGEConv,
}

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim = None, heads = 8, dim_head = 64, dropout = 0.5):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context = None, mask = None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h = h)
            sim.masked_fill_(~mask, max_neg_value)

        attn = sim.softmax(dim = -1)
        attn = self.dropout(attn)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.to_out(out)
    
class LMF(nn.Module):
    '''
    Low-rank Multimodal Fusion
    Credit to https://github.com/Justin1904/Low-rank-Multimodal-Fusion/blob/master/model.py
    '''

    def __init__(self, hidden_dims, output_dim, rank):
        '''
        Args:
            hidden_dims - another length-2 tuple, hidden dim of each graph network
            output_dim - int, specifying the size of output
            rank - int, specifying the size of rank in LMF
        '''
        super().__init__()

        self.x1_hidden = hidden_dims[0]
        self.x2_hidden = hidden_dims[1]
        self.output_dim = output_dim
        self.rank = rank

        self.x1_factor = nn.Parameter(torch.Tensor(self.rank, self.x1_hidden +1, self.output_dim))
        self.x2_factor = nn.Parameter(torch.Tensor(self.rank, self.x2_hidden +1, self.output_dim))
        self.fusion_weights = nn.Parameter(torch.Tensor(1, self.rank))
        self.fusion_bias = nn.Parameter(torch.Tensor(1, self.output_dim))

        # init the factors
        nn.init.xavier_normal_(self.x1_factor)
        nn.init.xavier_normal_(self.x2_factor)
        nn.init.xavier_normal_(self.fusion_weights)
        self.fusion_bias.data.fill_(0)

    def forward(self, x1, x2):
        '''
        Args:
            x1: tensor of shape (batch_size, x1_hidden_dim)
            x2: tensor of shape (batch_size, x2_hidden_dim)
        '''
        batch_size = x1.size(0)

        # next we perform low-rank multimodal fusion
        # here is a more efficient implementation than the one the paper describes
        # basically swapping the order of summation and elementwise product

        _x1 = torch.cat((torch.ones(batch_size, 1).to(x1.device), x1), dim=1)
        _x2 = torch.cat((torch.ones(batch_size, 1).to(x2.device), x2), dim=1)

        fusion_x1 = torch.matmul(_x1, self.x1_factor)
        fusion_x2 = torch.matmul(_x2, self.x2_factor)
        fusion_zy = fusion_x1 * fusion_x2

        # use linear transformation instead of simple summation, more flexibility
        output = torch.matmul(self.fusion_weights, fusion_zy.permute(1, 0, 2)).squeeze() + self.fusion_bias
        output = output.view(-1, self.output_dim)
        return output

def create_graph_dataset(cfg):
    #Load ids
    train_ids = pd.read_csv(cfg.data.train_ids, header=None, names=['id'])['id'].astype(str).values
    val_ids = pd.read_csv(cfg.data.val_ids, header=None, names=['id'])['id'].astype(str).values
    test_ids = pd.read_csv(cfg.data.test_ids, header=None, names=['id'])['id'].astype(str).values
    sample_ids = np.concatenate([train_ids, test_ids, val_ids])
    all_ids = pd.read_csv(cfg.data.labels)['id'].astype(str).values.tolist()
    # Load distance matrix and create adjacency matrix for the first graph network
    dist_df_1 = pd.read_csv(cfg.data.dist_matrix_1, index_col=0)
    mask_ids_1 = [True if col in all_ids else False for col in dist_df_1.columns]
    dist_df_1 = dist_df_1.copy().iloc[mask_ids_1, mask_ids_1]


    # TODO: implement randomly dropping edges

    if cfg.data.no_edge:
        adj_mat_1 = pd.DataFrame(np.zeros((dist_df_1.shape[0], dist_df_1.shape[0]), dtype=int),
                               index=dist_df_1.index,
                               columns=dist_df_1.columns)
    else:
        if not cfg.preexisting_adj_matrix.is_use:
            dist_list_1 = np.unique(dist_df_1.to_numpy().reshape(-1))
            dist_threshold_1 = 0
            if cfg.data.is_dist_1 == False: # Similarity matrix
                dist_list_1 = dist_list_1.copy()[::-1]

            total_num_edges_1 = dist_df_1.shape[0] * (dist_df_1.shape[0] - 1) / 2 
            edge_threshold_1 = int(total_num_edges_1 * (cfg.data.retaining_pmille / 1000))

            for i in dist_list_1:
                if cfg.data.is_dist_1:
                    num_edges_1 = ((dist_df_1 <= i).astype(int).sum().sum() - dist_df_1.shape[0]) / 2
                else:
                    num_edges_1 = ((dist_df_1 >= i).astype(int).sum().sum() - dist_df_1.shape[0]) / 2
                if num_edges_1 >= edge_threshold_1:
                    dist_threshold_1 = i
                    break
            
            assert dist_threshold_1 > 0, "Invalid distance threshold"
            if cfg.data.is_dist_1:
                adj_mat_1 = (dist_df_1 <= dist_threshold_1).astype(int)
            else:
                adj_mat_1 = (dist_df_1 >= dist_threshold_1).astype(int)

            # remove self-connected edges
            for i in range(adj_mat_1.shape[0]):
                adj_mat_1.iloc[i, i] = 0
        else:
            adj_mat_1 = pd.read_csv(cfg.preexisting_adj_matrix.file_path_1,
                                    index_col=0)
    mask_1 = [True if str(col) in sample_ids else False for col in adj_mat_1.columns.values]
    adj_mat_1 = adj_mat_1.copy().iloc[mask_1, mask_1]
    if cfg.data.use_decoupling:
        decoupling_mat = pd.read_csv(cfg.data.decoupling_matrix,
                                     header=0, index_col=0)
        decoupling_mat = decoupling_mat.copy().loc[adj_mat_1.columns, adj_mat_1.columns]
        adj_mat_1 = ((adj_mat_1 - decoupling_mat) > 0).astype(int)

    tmp_edge_index_1 = torch.Tensor(adj_mat_1.to_numpy()).nonzero().t() #convert to COO format

    #Load distance matrix and create adjacency matrix for the second graph network
    dist_df_2 = pd.read_csv(cfg.data.dist_matrix_2, index_col=0)
    mask_ids_2 = [True if col in all_ids else False for col in dist_df_2.columns]
    dist_df_2 = dist_df_2.copy().iloc[mask_ids_2, mask_ids_2]

    if cfg.data.no_edge:
        adj_mat_2 = pd.DataFrame(np.zeros((dist_df_2.shape[0], dist_df_2.shape[0]), dtype=int),
                               index=dist_df_2.index,
                               columns=dist_df_2.columns)
    else:
        if not cfg.preexisting_adj_matrix.is_use:
            dist_list_2 = np.unique(dist_df_2.to_numpy().reshape(-1))
            dist_threshold_2 = 0
            if cfg.data.is_dist_2 == False: # Similarity matrix
                dist_list_2 = dist_list_2.copy()[::-1]

            total_num_edges_2 = dist_df_2.shape[0] * (dist_df_2.shape[0] - 1) / 2 
            edge_threshold_2 = int(total_num_edges_2 * (cfg.data.retaining_pmille / 1000))

            for i in dist_list_2:
                if cfg.data.is_dist_2:
                    num_edges_2 = ((dist_df_2 <= i).astype(int).sum().sum() - dist_df_2.shape[0]) / 2
                else:
                    num_edges_2 = ((dist_df_2 >= i).astype(int).sum().sum() - dist_df_2.shape[0]) / 2
                if num_edges_2 >= edge_threshold_2:
                    dist_threshold_2 = i
                    break
            
            assert dist_threshold_2 > 0, "Invalid distance threshold"
            if cfg.data.is_dist_2:
                adj_mat_2 = (dist_df_2 <= dist_threshold_2).astype(int)
            else:
                adj_mat_2 = (dist_df_2 >= dist_threshold_2).astype(int)

            # remove self-connected edges
            for i in range(adj_mat_2.shape[0]):
                adj_mat_2.iloc[i, i] = 0
        else:
            adj_mat_2 = pd.read_csv(cfg.preexisting_adj_matrix.file_path_2,
                                    index_col=0)
            
        
    mask_2 = [True if col in sample_ids else False for col in adj_mat_2.columns]
    adj_mat_2 = adj_mat_2.copy().iloc[mask_2, mask_2]
    
    if cfg.data.use_decoupling:
        decoupling_mat = pd.read_csv(cfg.data.decoupling_matrix,
                                     header=0, index_col=0)
        decoupling_mat = decoupling_mat.copy().loc[adj_mat_2.columns, adj_mat_2.columns]
        adj_mat_2 = ((adj_mat_2 - decoupling_mat) > 0).astype(int)

    tmp_edge_index_2 = torch.Tensor(adj_mat_2.to_numpy()).nonzero().t() #convert to COO format

    # Sanity check for adj_mat_1 and adj_mat_2
    assert (adj_mat_1.columns == adj_mat_2.columns).sum() == len(adj_mat_1.columns), "Different columns in adj_mat_1 and adj_mat_2"
    assert (adj_mat_1.index == adj_mat_2.index).sum() == len(adj_mat_1.index), "Different index in adj_mat_1 and adj_mat_2"


    train_mask = torch.Tensor([True if col in train_ids else False for col in adj_mat_1.columns]).type(torch.BoolTensor)
    val_mask = torch.Tensor([True if col in val_ids else False for col in adj_mat_1.columns]).type(torch.BoolTensor)
    test_mask = torch.Tensor([True if col in test_ids else False for col in adj_mat_1.columns]).type(torch.BoolTensor)

    tmp_ids_df = pd.DataFrame({'id': adj_mat_1.columns})
    tmp_ids = torch.Tensor(tmp_ids_df.index.to_numpy()).type(torch.LongTensor)
    ast_df = pd.read_csv(cfg.data.labels)
    ast_df['id'] = ast_df['id'].astype(str)
    tmp_labels = torch.Tensor(tmp_ids_df.merge(ast_df, on='id')[cfg.data.antimicrobial]).type(torch.LongTensor)
    assert tmp_labels.size(0) == tmp_ids_df.shape[0], "Missing label value when merging"

    feats = []

    for isolate in adj_mat_1.columns:
        fp = os.path.join(cfg.data.input_dir, f"{isolate}.pt")
        tmp_feat = torch.load(fp).unsqueeze(dim=0)
        feats.append(tmp_feat)

    X = torch.cat(feats, dim=0)

    graph_data = geom_data.Data(x= X,
                                edge_index_1=tmp_edge_index_1,
                                edge_index_2=tmp_edge_index_2,
                                y=tmp_labels.unsqueeze(dim=-1),
                                train_mask=train_mask,
                                val_mask=val_mask,
                                test_mask=test_mask,
                                ids=tmp_ids,
                                isolate_codes=tmp_ids_df['id'].values)
    return graph_data, adj_mat_1, adj_mat_2

class GNNModel(nn.Module):
    def __init__(self, cfg, c_in, **kwargs):
        super().__init__()
        gnn_layer = gnn_layer_by_name[cfg.gnn.layer_name]
        in_channels = c_in
        out_channels = cfg.gnn.c_hidden
        self.middle_layer = cfg.gnn.middle_layer
        layers = []

        # Feature encoder
        if cfg.gnn.use_feature_encoder:
            layers += [
                nn.Linear(in_channels, out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(cfg.gnn.dp_rate)
            ]
            in_channels = out_channels

        # Graph neural net
        for layer_idx in range(cfg.gnn.num_layers):
            if cfg.gnn.layer_name in ["GAT", "GATv2", "TransformerConv"]:
                if layer_idx == (cfg.gnn.num_layers - 1):
                    kwargs['heads'] = 1
                    concat = False
                else:
                    concat = True

                layers += [
                    gnn_layer(in_channels=in_channels,
                              out_channels=out_channels,
                              concat=concat,
                              **kwargs),
                    nn.LayerNorm(out_channels * kwargs["heads"]),
                    nn.ELU(inplace=True),
                    nn.Dropout(cfg.gnn.dp_rate)
                ]
                in_channels = out_channels * kwargs["heads"]
            else:
                layers += [
                    gnn_layer(in_channels=in_channels,
                              out_channels=out_channels,
                              **kwargs),
                    nn.LayerNorm(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Dropout(cfg.gnn.dp_rate)
                ]
                in_channels = out_channels
        # layers += [nn.Linear(in_channels, cfg.gnn.c_out)]
        self.layers_1 = nn.ModuleList(layers)
        self.layers_2 = nn.ModuleList(layers)
        if cfg.gnn.middle_layer == "concat":
            self.middle_layers = nn.Linear(in_channels * 2, cfg.gnn.c_out)
        elif cfg.gnn.middle_layer in ["mean", "max"]:
            self.middle_layers = nn.Linear(in_channels, cfg.gnn.c_out)
        elif cfg.gnn.middle_layer == "simple":
            self.middle_layers = nn.Sequential(
                nn.Linear(in_channels*2, in_channels*2),
                nn.LayerNorm(in_channels*2),
                nn.ReLU(inplace=True),
                nn.Dropout(cfg.gnn.dp_rate),
                nn.Linear(in_channels * 2, cfg.gnn.c_out)
            )
        elif cfg.gnn.middle_layer == "cross_attn":
            self.middle_layers = CrossAttention(query_dim=in_channels, context_dim=in_channels, dropout=cfg.gnn.dp_rate)
            self.classification_head = nn.Linear(in_channels, cfg.gnn.c_out)
        elif cfg.gnn.middle_layer == "dual_cross_attn":
            self.middle_layers_1 = CrossAttention(query_dim=in_channels, context_dim=in_channels, dropout=cfg.gnn.dp_rate)
            self.middle_layers_2 = CrossAttention(query_dim=in_channels, context_dim=in_channels, dropout=cfg.gnn.dp_rate)
            self.classification_head = nn.Linear(in_channels*2, cfg.gnn.c_out)
        elif cfg.gnn.middle_layer == "low_rank_fusion":
            self.middle_layers = LMF(hidden_dims=(in_channels, in_channels), output_dim=cfg.gnn.c_out, rank=cfg.gnn.lmf.rank)

    def forward(self, x, edge_index_1, edge_index_2):
        x_1 = x.clone()
        x_2 = x.clone()
        for layer_1 in self.layers_1:
            if isinstance(layer_1, geom_nn.MessagePassing):
                x_1 = layer_1(x_1, edge_index_1)
            else:
                x_1 = layer_1(x_1)
        for layer_2 in self.layers_2:
            if isinstance(layer_2, geom_nn.MessagePassing):
                x_2 = layer_2(x_2, edge_index_2)
            else:
                x_2 = layer_2(x_2)

        if self.middle_layer == 'cross_attn':
            assert (len(x_1.size()) == 2) and (len(x_2.size()) == 2), "Invalid input shape for Cross Attention"
            x_1 = x_1.unsqueeze(dim=1) # (B, D) -> (B, 1, D)
            x_2 = x_2.unsqueeze(dim=1) # (B, D) -> (B, 1, D)
            x = self.middle_layers(x_1, context=x_2).squeeze(dim=1) # (B, 1, D) -> (B, D)
            x = self.classification_head(x)
        elif self.middle_layer == 'dual_cross_attn':
            assert (len(x_1.size()) == 2) and (len(x_2.size()) == 2), "Invalid input shape for Cross Attention"
            x_1 = x_1.unsqueeze(dim=1) # (B, D) -> (B, 1, D)
            x_2 = x_2.unsqueeze(dim=1) # (B, D) -> (B, 1, D)
            out_1 = self.middle_layers_1(x_1, context=x_2).squeeze(dim=1) # (B, 1, D) -> (B, D)
            out_2 = self.middle_layers_2(x_2, context=x_1).squeeze(dim=1) # (B, 1, D) -> (B, D)
            x = torch.cat([out_1, out_2], dim=-1)
            x = self.classification_head(x)
        elif self.middle_layer == "f":
            x = self.middle_layers(x_1, x_2)
        else:
            if self.middle_layer == "mean":
                x = torch.stack([x_1, x_2], dim=-1).mean(dim=-1)
            elif self.middle_layer == "max":
                x = torch.stack([x_1, x_2], dim=-1).max(dim=-1)
            else:
                x = torch.cat([x_1, x_2], dim=-1)
            x = self.middle_layers(x)
        return x
    
    
class AMRNodeDualGNN(L.LightningModule):
    def __init__(self, cfg, c_in, class_weight=None):
        super().__init__()
        self.save_hyperparameters(ignore=['cfg'])
        self.cfg = cfg
        self.auroc_fn = AUROC(task='binary')

        model_kwargs = {}

        if cfg.gnn.layer_name in ["GAT", "GATv2"]:
            model_kwargs['heads'] = cfg.gnn.GAT.heads
        elif cfg.gnn.layer_name == "TransformerConv":
            model_kwargs['heads'] = cfg.gnn.TransformerConv.heads

        self.model = GNNModel(cfg=cfg, c_in=c_in, **model_kwargs)

        if class_weight is not None:
            self.loss_fn = nn.BCELoss(class_weight)
        else:
            self.loss_fn = nn.BCELoss()

        # storing result of testing samples:
        self.test_results = {
            "isolate_ids": None,
            "targets": None,
            "proba_preds": None,
        }

    def forward(self, data, mode='train'):
        x, edge_index_1, edge_index_2 = data.x, data.edge_index_1, data.edge_index_2
        y_hat = self.model(x, edge_index_1, edge_index_2)
        y_hat = nn.Sigmoid()(y_hat)

        if mode == "train":
            mask = data.train_mask
        elif mode == "val":
            mask = data.val_mask
        elif mode == "test":
            mask = data.test_mask
        else:
            assert False, f"Unknown forward mode: {mode}"

        loss = self.loss_fn(y_hat[mask], data.y[mask].float())
        auroc = self.auroc_fn(y_hat[mask], data.y[mask])

        if mode == "test":
            print(f"x: {x.size()}; y_hat: {y_hat.size()};\t test_mask: {data.test_mask.size()}; ids: {data.ids.size()}")
            test_ids = data.ids[data.test_mask].cpu().numpy().astype(int)
            isolate_ids = np.array(data.isolate_codes).reshape(-1,)
            test_isolate_ids = isolate_ids.copy()[test_ids]
            test_proba = y_hat[data.test_mask].squeeze().cpu().numpy()
            test_targets = data.y[data.test_mask].squeeze().cpu().numpy()

            if self.test_results['targets'] is None:
                self.test_results['isolate_ids'] = test_isolate_ids
                self.test_results['targets'] = test_targets
                self.test_results['proba_preds'] = test_proba
            else:
                self.test_results['isolate_ids'] = np.concatenate([self.test_results['isolate_ids'], test_ids])
                self.test_results['targets'] = np.concatenate([self.test_results['targets'], test_targets])
                self.test_results['proba_preds'] = np.concatenate([self.test_results['proba_preds'], test_proba])

        return loss, auroc

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, 
                                                    milestones=[50, 100], gamma=0.1)
        return [optimizer], [scheduler]
    
    def training_step(self, batch, batch_idx):
        loss, auroc = self.forward(batch, mode="train")
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        self.log('train_auroc', auroc, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, auroc = self.forward(batch, mode="val")
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        self.log('val_auroc', auroc, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, auroc = self.forward(batch, mode="test")
        return loss

    def on_test_end(self):
        out_fp = os.path.join(self.cfg.trainer.model_checkpoint.dirpath, "test_results.csv")
        pd.DataFrame(self.test_results).to_csv(out_fp, index=False)

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg):
    L.seed_everything(cfg.trainer.seed * cfg.data.run_id)

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
    os.makedirs(cfg.trainer.model_checkpoint.dirpath,
                exist_ok=True)
    os.makedirs(cfg.trainer.logger.save_dir,
                exist_ok=True)

    # Saving adjacency matrix
    if cfg.data.save_adj_matrix:
        adj_mat_1.to_csv(os.path.join(cfg.trainer.model_checkpoint.dirpath, "adj_mat_1.csv"))
        adj_mat_2.to_csv(os.path.join(cfg.trainer.model_checkpoint.dirpath, "adj_mat_2.csv"))
    
    model_ckpt = ModelCheckpoint(
        dirpath=cfg.trainer.model_checkpoint.dirpath,
        mode=cfg.trainer.model_checkpoint.mode,
        monitor=cfg.trainer.model_checkpoint.monitor,
        save_last=cfg.trainer.model_checkpoint.save_last,
        save_top_k=cfg.trainer.model_checkpoint.save_top_k,
    )

    logger = WandbLogger(project=cfg.trainer.logger.project,
                         name=cfg.trainer.logger.name,
                         save_dir=cfg.trainer.logger.save_dir)

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
        logger=logger
    )

    # Training and save 
    trainer.fit(model,
                train_dataloaders=node_data_loader,
                val_dataloaders=node_data_loader)

    # Load the best checkpoint and evaluate on test set
    trainer.test(ckpt_path='best',
                 dataloaders=node_data_loader)
    
    # Clean up checkpoint files to save storage
    if cfg.trainer.del_ckpt:
        ckpt_files = glob.glob(os.path.join(cfg.trainer.model_checkpoint.dirpath, '*.ckpt'))

        for file in ckpt_files:
            os.remove(file)

if __name__ == "__main__":
    main()
