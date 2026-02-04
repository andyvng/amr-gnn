import argparse
import collections
import math
import os

import numpy as np
import pandas as pd
import scipy.spatial as spatial
from Bio import SeqIO

from utils import convert_distance_to_adjacency


def fcgr_transformation(sequence, k, max_kmers=0):
    """
    Chaos game representation
    Credit to https://towardsdatascience.com/chaos-game-representation-of-a-genetic-sequence-4681f1a67e14
    """

    seq = sequence.upper()

    kmer_count = collections.defaultdict(int)
    for i in range(len(seq) - (k - 1)):
        tmp_seq = seq[i : i + k]
        if not (("N" in tmp_seq) or ("-" in tmp_seq)):
            kmer_count[tmp_seq] += 1

    probabilities = collections.defaultdict(float)
    N = len(seq)
    for key, value in kmer_count.items():
        if max_kmers > 0:
            probabilities[key] = float(value) / max_kmers
        else:
            probabilities[key] = float(value) / (N - k + 1)

    array_size = int(math.sqrt(4**k))
    chaos = []
    for i in range(array_size):
        chaos.append([0] * array_size)
    maxx = array_size
    maxy = array_size
    posx = 1
    posy = 1

    for key, value in probabilities.items():
        for char in key:
            if char == "G":
                posx += maxx / 2
            elif char == "C":
                posy += maxy / 2
            elif char == "T":
                posx += maxx / 2
                posy += maxy / 2
            maxx = maxx / 2
            maxy /= 2
        chaos[int(posy - 1)][int(posx - 1)] = value
        maxx = array_size
        maxy: int = array_size
        posx = 1
        posy = 1

    return chaos


def get_args():
    parser = argparse.ArgumentParser(
        description="Compute distance matrix from FCGR features"
    )
    parser.add_argument(
        "--aligned_assembly_files", type=str, help="List of assembly file paths"
    )
    parser.add_argument("--position_file", type=str, help="AMR gene position file path")
    parser.add_argument("--kmer_size", type=int, help="k-mer size for FCGR computation")
    parser.add_argument(
        "--contig_id", type=str, help="Contig ID to extract from assembly"
    )
    parser.add_argument(
        "--keep_proportion", type=float, help="Proportion of edges to keep"
    )
    parser.add_argument(
        "--output", type=str, help="Output file path for distance matrix"
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    positions = pd.read_csv(args.position_file, header=0)
    k = args.kmer_size
    max_kmers = np.max(positions["end"] - positions["start"] - k + 2)

    feat_table = []
    ids = []
    aligned_assembly_files = args.aligned_assembly_files.split(",")

    for assembly_file in aligned_assembly_files:
        isolate_id = os.path.basename(os.path.dirname(assembly_file))
        ids.append(isolate_id)
        sequences = []

        # Change to SeqIO.parse if multiple sequences in a file
        records = SeqIO.parse(assembly_file, "fasta")

        for tmp_record in records:
            if tmp_record.id == args.contig_id:
                curr_record = tmp_record
                break

        for _, row in positions.iterrows():
            start_pos = row["start"]
            end_pos = row["end"]
            tmp_seq = curr_record.seq[start_pos - 1 : end_pos].__str__().strip().upper()
            chaos_seq = fcgr_transformation(tmp_seq, k, max_kmers=max_kmers)
            sequences.append(chaos_seq)

        tmp_feat = np.stack(sequences, axis=0)
        feat_table.append(tmp_feat)

    feat_table = np.stack(feat_table, axis=0)
    feat_table = feat_table.reshape(feat_table.shape[0], -1)
    dist_array = spatial.distance.cdist(feat_table, feat_table, metric="euclidean")

    dist_matrix_df = pd.DataFrame(dist_array, index=ids, columns=ids)

    convert_distance_to_adjacency(
        dist_matrix_df, keep_proportion=args.keep_proportion
    ).to_csv(args.output)


if __name__ == "__main__":
    main()
