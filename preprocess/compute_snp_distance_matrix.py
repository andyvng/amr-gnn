import argparse
import os

import numpy as np
import pandas as pd
import scipy.spatial as spatial

from utils import convert_distance_to_adjacency


def get_args():
    parser = argparse.ArgumentParser(
        description="Compute distance matrix from SNP matrix"
    )
    parser.add_argument(
        "--vcf_files",
        type=str,
        help="VCF file paths with comma-separated format",
    )
    parser.add_argument(
        "--keep_proportion", type=float, help="Proportion of edges to keep"
    )
    parser.add_argument(
        "--out_fp", type=str, help="Output file path for distance matrix"
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    dfs = []

    vcf_files = args.vcf_files.split(",")

    for vcf_file in vcf_files:
        tmp_df = pd.read_csv(
            vcf_file, header=0, sep="\t", usecols=["POS", "ALT", "EFFECT"]
        )
        isolate_id = os.path.basename(os.path.dirname(vcf_file))

        # Remove synonymous SNPs in the EFFECTs columns
        tmp_df = tmp_df[
            ~tmp_df["EFFECT"].str.contains("synonymous_variant", na=False)
        ].reset_index(drop=True)
        tmp_df["isolate_id"] = isolate_id
        tmp_df["POS"] = tmp_df["POS"].astype(int)
        tmp_df["ALT"] = tmp_df["ALT"].astype(str)
        tmp_df.drop(columns=["EFFECT"], inplace=True)
        dfs.append(tmp_df)

    snp_df = pd.concat(dfs, axis=0, ignore_index=True)
    matrix_df = snp_df.pivot(index="isolate_id", columns="POS", values="ALT")
    matrix_df = matrix_df.fillna("REF")

    encoded_matrix = matrix_df[matrix_df.columns].apply(
        lambda x: x.astype("category").cat.codes
    )
    dist_array = spatial.distance.cdist(
        encoded_matrix, encoded_matrix, metric="hamming"
    )
    dist_matrix_df = pd.DataFrame(
        dist_array, index=matrix_df.index, columns=matrix_df.index
    )

    convert_distance_to_adjacency(
        dist_matrix_df, keep_proportion=args.keep_proportion
    ).to_csv(args.out_fp)


if __name__ == "__main__":
    main()
