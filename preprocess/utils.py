import os

import numpy as np
import pandas as pd


def convert_distance_to_adjacency(
    dist_df, keep_proportion=None, threshold=None, convert_to_COO=False
):
    if keep_proportion is not None:  # TODO: need to check this
        # Total possible edges = n*(n-1)/2
        total_edges = dist_df.shape[0] * (dist_df.shape[0] - 1) / 2
        # Edges to keep based on proportion
        edges_to_keep = int(total_edges * keep_proportion)

        # find distance threshold
        for threshold in np.unique(dist_df.to_numpy().reshape(-1)):
            num_edges = (
                (dist_df <= threshold).astype(int).sum().sum() - dist_df.shape[0]
            ) / 2
            dist_threshold = threshold
            if num_edges >= edges_to_keep:
                break
        adj_matrix = (dist_df <= dist_threshold).astype(int)
    elif threshold is not None:
        adj_matrix = (dist_df <= threshold).astype(int)
    else:
        raise ValueError("Either keep_proportion or threshold must be provided.")

    # remove self-connected edges
    for i in range(adj_matrix.shape[0]):
        adj_matrix.iloc[i, i] = 0

    if convert_to_COO:  # TODO: need to check this
        adj_matrix = adj_matrix.to_numpy()
        row, col = np.where(adj_matrix == 1)
        data = np.ones(len(row), dtype=int)
        adj_matrix = (row, col, data)

    return adj_matrix
