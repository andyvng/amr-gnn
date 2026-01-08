import os

import lightning as L
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch_geometric.nn as geom_nn
from einops import rearrange, repeat
from torch import einsum, nn
from torchmetrics import AUROC

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
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.5):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head**-0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))

        sim = einsum("b i d, b j d -> b i j", q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, "b ... -> b (...)")
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, "b j -> (b h) () j", h=h)
            sim.masked_fill_(~mask, max_neg_value)

        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)

        out = einsum("b i j, b j d -> b i d", attn, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        return self.to_out(out)


class LMF(nn.Module):
    """
    Low-rank Multimodal Fusion
    Credit to https://github.com/Justin1904/Low-rank-Multimodal-Fusion/blob/master/model.py
    """

    def __init__(self, hidden_dims, output_dim, rank):
        """
        Args:
            hidden_dims - another length-2 tuple, hidden dim of each graph network
            output_dim - int, specifying the size of output
            rank - int, specifying the size of rank in LMF
        """
        super().__init__()

        self.x1_hidden = hidden_dims[0]
        self.x2_hidden = hidden_dims[1]
        self.output_dim = output_dim
        self.rank = rank

        self.x1_factor = nn.Parameter(
            torch.Tensor(self.rank, self.x1_hidden + 1, self.output_dim)
        )
        self.x2_factor = nn.Parameter(
            torch.Tensor(self.rank, self.x2_hidden + 1, self.output_dim)
        )
        self.fusion_weights = nn.Parameter(torch.Tensor(1, self.rank))
        self.fusion_bias = nn.Parameter(torch.Tensor(1, self.output_dim))

        # init the factors
        nn.init.xavier_normal_(self.x1_factor)
        nn.init.xavier_normal_(self.x2_factor)
        nn.init.xavier_normal_(self.fusion_weights)
        self.fusion_bias.data.fill_(0)

    def forward(self, x1, x2):
        """
        Args:
            x1: tensor of shape (batch_size, x1_hidden_dim)
            x2: tensor of shape (batch_size, x2_hidden_dim)
        """
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
        output = (
            torch.matmul(self.fusion_weights, fusion_zy.permute(1, 0, 2)).squeeze()
            + self.fusion_bias
        )
        output = output.view(-1, self.output_dim)
        return output


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
                nn.Dropout(cfg.gnn.dp_rate),
            ]
            in_channels = out_channels

        # Graph neural net
        for layer_idx in range(cfg.gnn.num_layers):
            if cfg.gnn.layer_name in ["GAT", "GATv2", "TransformerConv"]:
                if layer_idx == (cfg.gnn.num_layers - 1):
                    kwargs["heads"] = 1
                    concat = False
                else:
                    concat = True

                layers += [
                    gnn_layer(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        concat=concat,
                        **kwargs,
                    ),
                    nn.LayerNorm(out_channels * kwargs["heads"]),
                    nn.ELU(inplace=True),
                    nn.Dropout(cfg.gnn.dp_rate),
                ]
                in_channels = out_channels * kwargs["heads"]
            else:
                layers += [
                    gnn_layer(
                        in_channels=in_channels, out_channels=out_channels, **kwargs
                    ),
                    nn.LayerNorm(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Dropout(cfg.gnn.dp_rate),
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
                nn.Linear(in_channels * 2, in_channels * 2),
                nn.LayerNorm(in_channels * 2),
                nn.ReLU(inplace=True),
                nn.Dropout(cfg.gnn.dp_rate),
                nn.Linear(in_channels * 2, cfg.gnn.c_out),
            )
        elif cfg.gnn.middle_layer == "cross_attn":
            self.middle_layers = CrossAttention(
                query_dim=in_channels, context_dim=in_channels, dropout=cfg.gnn.dp_rate
            )
            self.classification_head = nn.Linear(in_channels, cfg.gnn.c_out)
        elif cfg.gnn.middle_layer == "dual_cross_attn":
            self.middle_layers_1 = CrossAttention(
                query_dim=in_channels, context_dim=in_channels, dropout=cfg.gnn.dp_rate
            )
            self.middle_layers_2 = CrossAttention(
                query_dim=in_channels, context_dim=in_channels, dropout=cfg.gnn.dp_rate
            )
            self.classification_head = nn.Linear(in_channels * 2, cfg.gnn.c_out)
        elif cfg.gnn.middle_layer == "low_rank_fusion":
            self.middle_layers = LMF(
                hidden_dims=(in_channels, in_channels),
                output_dim=cfg.gnn.c_out,
                rank=cfg.gnn.lmf.rank,
            )

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

        if self.middle_layer == "cross_attn":
            assert (len(x_1.size()) == 2) and (len(x_2.size()) == 2), (
                "Invalid input shape for Cross Attention"
            )
            x_1 = x_1.unsqueeze(dim=1)  # (B, D) -> (B, 1, D)
            x_2 = x_2.unsqueeze(dim=1)  # (B, D) -> (B, 1, D)
            x = self.middle_layers(x_1, context=x_2).squeeze(
                dim=1
            )  # (B, 1, D) -> (B, D)
            x = self.classification_head(x)
        elif self.middle_layer == "dual_cross_attn":
            assert (len(x_1.size()) == 2) and (len(x_2.size()) == 2), (
                "Invalid input shape for Cross Attention"
            )
            x_1 = x_1.unsqueeze(dim=1)  # (B, D) -> (B, 1, D)
            x_2 = x_2.unsqueeze(dim=1)  # (B, D) -> (B, 1, D)
            out_1 = self.middle_layers_1(x_1, context=x_2).squeeze(
                dim=1
            )  # (B, 1, D) -> (B, D)
            out_2 = self.middle_layers_2(x_2, context=x_1).squeeze(
                dim=1
            )  # (B, 1, D) -> (B, D)
            x = torch.cat([out_1, out_2], dim=-1)
            x = self.classification_head(x)
        elif self.middle_layer == "low_rank_fusion":
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
        self.save_hyperparameters(ignore=["cfg"])
        self.cfg = cfg
        self.auroc_fn = AUROC(task="binary")

        model_kwargs = {}

        if cfg.gnn.layer_name in ["GAT", "GATv2"]:
            model_kwargs["heads"] = cfg.gnn.GAT.heads
        elif cfg.gnn.layer_name == "TransformerConv":
            model_kwargs["heads"] = cfg.gnn.TransformerConv.heads

        self.model = GNNModel(cfg=cfg, c_in=c_in, **model_kwargs)

        if class_weight is not None:
            self.loss_fn = nn.BCELoss(class_weight)
        else:
            self.loss_fn = nn.BCELoss()

        # storing result of testing samples:
        self.results = {
            "isolate_id": None,
            "y_true": None,
            "y_proba": None,
        }

    def forward(self, data, mode="train"):
        x, edge_index_1, edge_index_2 = data.x, data.edge_index_1, data.edge_index_2
        y_hat = self.model(x, edge_index_1, edge_index_2)
        y_hat = nn.Sigmoid()(y_hat)

        if mode == "train":
            mask = data.train_mask
        elif mode == "val":
            mask = data.val_mask
        elif mode == "test":
            mask = data.test_mask
        elif mode == "predict":
            mask = data.predict_mask
        else:
            assert False, f"Unknown forward mode: {mode}"

        loss = self.loss_fn(y_hat[mask], data.y[mask].float())
        auroc = self.auroc_fn(y_hat[mask], data.y[mask])

        if mode in ["test", "predict"]:
            tmp_ids = data.ids[mask].cpu().numpy().astype(int)
            isolate_ids = np.array(data.isolate_codes).reshape(
                -1,
            )
            tmp_isolate_ids = isolate_ids.copy()[tmp_ids]
            tmp_proba = y_hat[mask].squeeze().cpu().numpy()
            tmp_targets = data.y[mask].squeeze().cpu().numpy()

            if self.results["y_true"] is None:
                self.results["isolate_id"] = tmp_isolate_ids
                self.results["y_true"] = tmp_targets
                self.results["y_proba"] = tmp_proba
            else:
                self.results["isolate_id"] = np.concatenate(
                    [self.results["isolate_id"], tmp_isolate_ids]
                )
                self.results["y_true"] = np.concatenate(
                    [self.results["y_true"], tmp_targets]
                )
                self.results["y_proba"] = np.concatenate(
                    [self.results["y_proba"], tmp_proba]
                )
        return loss, auroc

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[50, 100], gamma=0.1
        )
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        loss, auroc = self.forward(batch, mode="train")
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        self.log("train_auroc", auroc, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, auroc = self.forward(batch, mode="val")
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.log("val_auroc", auroc, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, auroc = self.forward(batch, mode="test")
        return loss

    def _generate_results(self, df, out_fp):
        df["y_pred"] = df["y_proba"].apply(
            lambda x: "Resistant" if x > 0.5 else "Susceptible"
        )
        df["y_true"] = df["y_true"].apply(
            lambda x: "Resistant" if x == 1 else "Susceptible"
        )
        rename_dict = {
            "y_true": f"True AST ({self.cfg.data.antimicrobial.capitalize()})",
            "y_pred": f"Predicted AST ({self.cfg.data.antimicrobial.capitalize()})",
            "y_proba": "Predicted probability",
        }
        df.rename(columns=rename_dict, inplace=True)
        df.to_csv(out_fp, index=False)

    def on_test_end(self):
        out_fp = os.path.join(
            self.cfg.trainer.model_checkpoint.dirpath, "test_results.csv"
        )
        result_df = pd.DataFrame(self.results)
        self._generate_results(result_df, out_fp)
        # result_df["y_pred"] = result_df["y_proba"].apply(
        #     lambda x: "resistant" if x > 0.5 else "susceptible"
        # )
        # result_df.to_csv(out_fp, index=False)

    def predict_step(self, batch, batch_idx):
        loss, auroc = self.forward(batch, mode="predict")
        return loss

    def on_predict_end(self):
        out_fp = os.path.join(self.cfg.prediction.outdir, "prediction_results.csv")
        result_df = pd.DataFrame(self.results)
        self._generate_results(result_df, out_fp)
        # result_df["y_pred"] = result_df["y_proba"].apply(
        #     lambda x: "resistant" if x > 0.5 else "susceptible"
        # )
        # result_df.to_csv(out_fp, index=False)
