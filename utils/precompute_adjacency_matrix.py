import os
import pandas as pd
import numpy as np
import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Get the selected kmers')
    parser.add_argument('dist_fp', type=str, help='Distance matrix file path')
    parser.add_argument('antimicrobial', type=str, help='Antimicrobial drug')
    parser.add_argument('label_fp', type=str, help='File path to AST data')
    parser.add_argument('outdir', type=str, help="Output file path")
    parser.add_argument('--min', type=int, default=2, help='Minimum threshold')
    parser.add_argument('--max', type=int, default=21, help='Max threshold')
    parser.add_argument('--step', type=int, default=2, help='Step size')
    args = parser.parse_args()
    return args

def main():
    args = get_args()

    dist_df = pd.read_csv(args.dist_fp, header=0, index_col=0)
    ast_df = pd.read_csv(args.label_fp, header=0)
    ids = ast_df[ast_df[args.antimicrobial].notnull()]['id'].astype(str).values
    id_mask = [True if col in ids else False for col in dist_df.columns]
    dist_df = dist_df.copy().iloc[id_mask, id_mask]

    total_edges = dist_df.shape[0]*(dist_df.shape[0] - 1) / 2
    dist_list = np.unique(dist_df.to_numpy().reshape(-1))

    tmp_outdir = os.path.join(args.outdir, args.antimicrobial)
    os.makedirs(tmp_outdir, exist_ok=True)

    for threshold in range(args.min, args.max, args.step):
        edge_threshold = int(total_edges * (threshold / 1000))

        for i in dist_list:
            num_edges = ((dist_df <= i).astype(int).sum().sum() - dist_df.shape[0]) / 2
            
            if num_edges >= edge_threshold:
                criterion = i
                break

        adj_mat = (dist_df <= criterion).astype(int)

        for i in range(adj_mat.shape[0]):
            adj_mat.iloc[i, i]=0
        
        adj_mat.to_csv(os.path.join(tmp_outdir, f"permille_{threshold}.csv"))

if __name__ == "__main__":
    main()