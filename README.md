# AMR-GNN: A multi-representation graph neural network framework to enable genomic antimicrobial resistance prediction

This is the code repository of the [paper](https://www.biorxiv.org/)

## Overview of AMR-GNN Workflow
![plot](./assets/workflow.png)

## Installation
We use `uv` to manage Python environment and install tools used for this work. Please refer [here](https://docs.astral.sh/uv/getting-started/installation/) to install `uv`.

Install essential tools:

```bash
uv tool add gdown snakemake
```

## Docker images
We use Docker to containerise AMR-GNN codebase. First, clone this repository and navigate to the project directory.

```bash
git clone https://github.com/andyvng/amr-gnn.git && \
cd amr-gnn
```

Build the Docker image:

```bash
docker build -f Dockerfile -t amrgnn:latest .
```

## Data preprocessing
We present a Snakemake workflow designed to generate feature input data from genome assemblies. First, download the following prerequisites from Google Drive. These files include the unitig database, the list of selected unitigs, the PAO1 reference genome, and a BED file marking curated *P. aeruginosa* AMR genes.

```bash
gdown [URL] -O output/path --folder
```

To execute the workflow, place all genome assemblies into a single directory and specify the file extension (default: .fasta). The workflow automatically identifies all assemblies within the directory to generate the corresponding node features and adjacency matrices.

```
cd preprocess
snakemake --cores 'all' assembly_dir="assemblies" \
                        unitig_db="unitig_db" \
                        ref_genome="ref_genome" \
                        amr_positions="amr_positions"

```

## Train AMR-GNN model
As an example to train AMR-GNN and perform AMR prediction afterwards, we provide real-world data (BV-BRC database) for predicting vancomycin resistance in <em>Enterococcus faecium</em>. Please download the dataset and mount this `data` folder to the Docker container to perform training.

```bash
gdown URL -O . --folder 
```

The `data` folder contains the AST label files, the ids of E. faecium isoaltes included in the study, the selected unitigs for node features, two adjancency matrices derived from SNPS and FCGR features. Here is the structure of the `data` folder

<p align="center">
  <img src="./assets/file_tree.png" alt="Tree file" width="300">
</p>


Train AMR-GNN for predicting vancomycin resistance

```
docker run --rm \
           -v ${PWD}/data:/amrgnn/data \
           -v ${PWD}/experiments:/amrgnn/experiments \
           amrgnn:latest \
           src/train.py \
           data.antimicrobial='vancomycin' \
           data.labels=data/ast_labels.csv \
           data.whole_ids=data/whole.ids \
           data.train_ids=data/train.ids \
           data.val_ids=data/val.ids \
           data.test_ids=data/predict.ids \
           adj_matrix.file_path_1=data/fcgr_adj_matrix.csv \
           adj_matrix.file_path_2=data/snps_adj_matrix.csv \
           trainer.model_checkpoint.dirpath=experiments/checkpoints 
```

Here, we used [Hydra](https://hydra.cc) to manage configuration settings. For the full list of configuration settings, please see [here](./conf/config.yaml)

## Predict AMR phenotype
Load trained model to predict vancomycin resistance.
```
docker run --rm \
           -v ${PWD}/data:/amrgnn/data \
           -v ${PWD}/experiments:/amrgnn/experiments \
           amrgnn:latest \
           src/predict.py \
           data.antimicrobial='vancomycin' \
           data.labels=data/ast_labels.csv \
           data.whole_ids=data/whole.ids \
           data.predict_ids=data/predict.ids \
           adj_matrix.file_path_1=data/fcgr_adj_matrix.csv \
           adj_matrix.file_path_2=data/snps_adj_matrix.csv \
           prediction.outdir="experiments/results"
```

The output files (`prediction_results.csv`) is a csv file include 4 columns
- The isolate id
- The true AMR phenotype (Resistant/Susceptible)
- The predicted probability from AMR-GNN
- The predicted AMR phenotype (Resistant/Susceptible)

## Model interpretation

We used Integrated gradients identify salient features. To compute the IG for each feature, please run the folowing command 

```
uv run interpretaion.py
```

The output include the the IG for each features of each isolates.


## Citation
```
@article {Nguyen2025.07.24.666581,
	author = {Nguyen, Hoai-An and Peleg, Anton Y. and Wisniewski, Jessica A. and Wang, Xiaoyu and Wang, Zhikang and Blakeway, Luke V. and Badoordeen, Gnei Z. and Theegala, Ravali and Doan, Nhu Quynh and Parker, Matthew H. and Green, Anna G. and Song, Jiangning and Dowe, David L. and Macesic, Nenad},
	title = {AMR-GNN: A multi-representation graph neural network framework to enable genomic antimicrobial resistance prediction},
	year = {2025},
	doi = {10.1101/2025.07.24.666581},
	URL = {https://www.biorxiv.org/content/early/2025/07/27/2025.07.24.666581},
	journal = {bioRxiv}
}
```