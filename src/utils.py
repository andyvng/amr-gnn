import os

import numpy as np
import pandas as pd
import torch
from torch_geometric import data as geom_data


def create_graph_dataset(cfg):
    # Load ids
    if os.path.exists(cfg.data.train_ids):
        train_ids = (
            pd.read_csv(cfg.data.train_ids, header=None, names=["id"])["id"]
            .astype(str)
            .values
        )
    if os.path.exists(cfg.data.val_ids):
        val_ids = (
            pd.read_csv(cfg.data.val_ids, header=None, names=["id"])["id"]
            .astype(str)
            .values
        )
    if os.path.exists(cfg.data.test_ids):
        test_ids = (
            pd.read_csv(cfg.data.test_ids, header=None, names=["id"])["id"]
            .astype(str)
            .values
        )
    if os.path.exists(cfg.data.predict_ids):
        predict_ids = (
            pd.read_csv(cfg.data.predict_ids, header=None, names=["id"])["id"]
            .astype(str)
            .values
        )
    sample_ids = (
        pd.read_csv(cfg.data.whole_ids, header=None, names=["id"])["id"]
        .astype(str)
        .values
    )
    all_ids = pd.read_csv(cfg.data.labels)["id"].astype(str).values.tolist()

    if cfg.adj_matrix.use_precompute_matrix:
        adj_mat_1: pd.DataFrame = pd.read_csv(cfg.adj_matrix.file_path_1, index_col=0)
        adj_mat_2 = pd.read_csv(cfg.adj_matrix.file_path_2, index_col=0)
    else:
        # Load distance matrix and create adjacency matrix for the first graph network
        dist_df_1 = pd.read_csv(cfg.adj_matrix.file_path_1, index_col=0)
        mask_ids_1 = [True if col in all_ids else False for col in dist_df_1.columns]
        dist_df_1 = dist_df_1.copy().iloc[mask_ids_1, mask_ids_1]

        dist_df_2 = pd.read_csv(cfg.adj_matrix.file_path_2, index_col=0)
        mask_ids_2 = [True if col in all_ids else False for col in dist_df_2.columns]
        dist_df_2 = dist_df_2.copy().iloc[mask_ids_2, mask_ids_2]

        if cfg.data.no_edge:
            adj_mat_1 = pd.DataFrame(
                np.zeros((dist_df_1.shape[0], dist_df_1.shape[0]), dtype=int),
                index=dist_df_1.index,
                columns=dist_df_1.columns,
            )

            adj_mat_2 = pd.DataFrame(
                np.zeros((dist_df_2.shape[0], dist_df_2.shape[0]), dtype=int),
                index=dist_df_2.index,
                columns=dist_df_2.columns,
            )
        else:
            dist_list_1 = np.unique(dist_df_1.to_numpy().reshape(-1))
            dist_threshold_1 = 0

            total_num_edges_1 = dist_df_1.shape[0] * (dist_df_1.shape[0] - 1) / 2
            edge_threshold_1 = int(
                total_num_edges_1 * (cfg.data.retaining_pmille / 1000)
            )

            for i in dist_list_1:
                num_edges_1 = (
                    (dist_df_1 <= i).astype(int).sum().sum() - dist_df_1.shape[0]
                ) / 2
                if num_edges_1 >= edge_threshold_1:
                    dist_threshold_1 = i
                    break

            assert dist_threshold_1 > 0, "Invalid distance threshold"
            adj_mat_1 = (dist_df_1 <= dist_threshold_1).astype(int)

            # remove self-connected edges
            for i in range(adj_mat_1.shape[0]):
                adj_mat_1.iloc[i, i] = 0

            dist_list_2 = np.unique(dist_df_2.to_numpy().reshape(-1))
            dist_threshold_2 = 0
            total_num_edges_2 = dist_df_2.shape[0] * (dist_df_2.shape[0] - 1) / 2
            edge_threshold_2 = int(
                total_num_edges_2 * (cfg.data.retaining_pmille / 1000)
            )

            for i in dist_list_2:
                num_edges_2 = (
                    (dist_df_2 <= i).astype(int).sum().sum() - dist_df_2.shape[0]
                ) / 2
                if num_edges_2 >= edge_threshold_2:
                    dist_threshold_2 = i
                    break

            assert dist_threshold_2 > 0, "Invalid distance threshold"
            adj_mat_2 = (dist_df_2 <= dist_threshold_2).astype(int)

            # remove self-connected edges
            for i in range(adj_mat_2.shape[0]):
                adj_mat_2.iloc[i, i] = 0

    mask_1 = [
        True if str(col) in sample_ids else False for col in adj_mat_1.columns.values
    ]
    adj_mat_1 = adj_mat_1.copy().iloc[mask_1, mask_1]
    if cfg.data.use_decoupling:
        decoupling_mat = pd.read_csv(cfg.data.decoupling_matrix, header=0, index_col=0)
        decoupling_mat = decoupling_mat.copy().loc[adj_mat_1.columns, adj_mat_1.columns]
        adj_mat_1 = ((adj_mat_1 - decoupling_mat) > 0).astype(int)

    tmp_edge_index_1 = (
        torch.Tensor(adj_mat_1.to_numpy()).nonzero().t()
    )  # convert to COO format

    mask_2 = [True if col in sample_ids else False for col in adj_mat_2.columns]
    adj_mat_2 = adj_mat_2.copy().iloc[mask_2, mask_2]

    if cfg.data.use_decoupling:
        decoupling_mat = pd.read_csv(cfg.data.decoupling_matrix, header=0, index_col=0)
        decoupling_mat = decoupling_mat.copy().loc[adj_mat_2.columns, adj_mat_2.columns]
        adj_mat_2 = ((adj_mat_2 - decoupling_mat) > 0).astype(int)

    tmp_edge_index_2 = (
        torch.Tensor(adj_mat_2.to_numpy()).nonzero().t()
    )  # convert to COO format

    # Sanity check for adj_mat_1 and adj_mat_2
    assert (adj_mat_1.columns == adj_mat_2.columns).sum() == len(adj_mat_1.columns), (
        "Different columns in adj_mat_1 and adj_mat_2"
    )
    assert (adj_mat_1.index == adj_mat_2.index).sum() == len(adj_mat_1.index), (
        "Different index in adj_mat_1 and adj_mat_2"
    )

    # Create masks
    if os.path.exists(cfg.data.train_ids):
        train_mask = torch.Tensor(
            [True if col in train_ids else False for col in adj_mat_1.columns]
        ).type(torch.BoolTensor)
    else:
        train_mask = None
    if os.path.exists(cfg.data.val_ids):
        val_mask = torch.Tensor(
            [True if col in val_ids else False for col in adj_mat_1.columns]
        ).type(torch.BoolTensor)
    else:
        val_mask = None
    if os.path.exists(cfg.data.test_ids):
        test_mask = torch.Tensor(
            [True if col in test_ids else False for col in adj_mat_1.columns]
        ).type(torch.BoolTensor)
    else:
        test_mask = None
    if os.path.exists(cfg.data.predict_ids):
        predict_mask = torch.Tensor(
            [True if col in predict_ids else False for col in adj_mat_1.columns]
        ).type(torch.BoolTensor)
    else:
        predict_mask = None

    tmp_ids_df = pd.DataFrame({"id": adj_mat_1.columns})
    tmp_ids = torch.Tensor(tmp_ids_df.index.to_numpy()).type(torch.LongTensor)
    ast_df = pd.read_csv(cfg.data.labels)
    ast_df["id"] = ast_df["id"].astype(str)
    tmp_labels = torch.Tensor(
        tmp_ids_df.merge(ast_df, on="id")[cfg.data.antimicrobial]
    ).type(torch.LongTensor)
    assert tmp_labels.size(0) == tmp_ids_df.shape[0], "Missing label value when merging"

    feats = []

    for isolate in adj_mat_1.columns:
        fp = os.path.join(cfg.data.input_dir, f"{isolate}.pt")
        tmp_feat = torch.load(fp).unsqueeze(dim=0)
        feats.append(tmp_feat)

    X = torch.cat(feats, dim=0)

    graph_data = geom_data.Data(
        x=X,
        edge_index_1=tmp_edge_index_1,
        edge_index_2=tmp_edge_index_2,
        y=tmp_labels.unsqueeze(dim=-1),
        train_mask=train_mask,
        val_mask=val_mask,
        predict_mask=predict_mask,
        test_mask=test_mask,
        ids=tmp_ids,
        isolate_codes=tmp_ids_df["id"].values,
    )
    return graph_data, adj_mat_1, adj_mat_2
