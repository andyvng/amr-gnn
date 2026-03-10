import argparse
import os

import numpy as np
import pandas as pd
import torch


def get_args():
    parser = argparse.ArgumentParser(
        description="Convert unitig presence/absence to node features"
    )
    parser.add_argument(
        "--unitig_query_results", type=str, help="Results from unitig-caller query"
    )
    parser.add_argument(
        "--unitig_list", type=str, help="File path to the reference unitig list"
    )
    parser.add_argument("--isolate_ids", type=str, help="Isolate IDs")
    parser.add_argument("--outdir", type=str, help="Output directory")
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    os.makedirs(args.outdir, exist_ok=True)
    isolate_ids = [isolate_id.strip() for isolate_id in args.isolate_ids.split(",")]

    ref_unitigs = pd.read_csv(args.unitig_list, header=0)

    for isolate_id in isolate_ids:
        # Extract lines from unitig query results containing the isolate ID
        isolate_results = []
        with open(args.unitig_query_results, "r") as f:
            for line in f:
                if isolate_id in line:
                    isolate_results.append(line.split("|")[0].strip())

        isolate_unitigs = pd.DataFrame(isolate_results, columns=["unitig"])
        merged_df = ref_unitigs.merge(
            isolate_unitigs, on="unitig", how="left", indicator=True
        )
        merged_df["presence"] = np.where(merged_df["_merge"] == "both", 1, 0)

        # Convert to tensor and save as .pt file
        ref_feats = torch.Tensor(merged_df["presence"])
        torch.save(ref_feats, os.path.join(args.outdir, f"{isolate_id}.pt"))


if __name__ == "__main__":
    main()
