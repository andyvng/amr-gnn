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

## Apptainer (formerly Singularity) images
If you are using Apptainer, it is possible to convert the Docker image into Apptainer image.
```
apptainer build amrgnn.sif docker-daemon://amrgnn:latest
```

## Data preprocessing
We provide a Snakemake workflow for generating feature inputs from genome assemblies. You will need to provide the list of selected unitigs from Elastic Net, an annotated reference genome for variant calling, and a CSV file detailing the positions of curated AMR genes on the reference genome. To execute the workflow, place all genome assemblies into a single directory and ensure the file extension is correctly specified (default: `.fasta`). The pipeline will automatically detect all assemblies within the target directory and generate the corresponding node features and adjacency matrices.

The following files need to be specified in the [config file](./preprocess/config.yaml) to run the pipeline:

| Argument                               | Description | 
|----------------------------------------|-------------|
| assemblies_dir  | Input directory containing genome assemblies |
| reference_genome | Annotated reference genome (e.g., PAO1 genome) |
| unitig_list | Unitig list (with header) selected by Pyseer's elastic net. One unitig per line |
| position_fp | Position of selected AMR genes on the reference genome for FCGR feature extraction |
| keep_proportion | Retention proportion based on the total number of possible edges |

Run the pipeline

```
cd preprocess
snakemake --cores 'all'
```

## Train AMR-GNN model
As an example to train AMR-GNN and perform AMR prediction afterwards, we provide real-world data (BV-BRC database) for predicting vancomycin resistance in <em>Enterococcus faecium</em>. Please download the dataset and mount this `data` folder to the Docker container to perform training.

```bash
gdown 1KZlLvAF8jPyPk2YkW-LNbqb0gFUmu7rS
tar -xzvf data.tar.gz && rm data.tar.gz
```

The `data` folder contains the AST label files, the ids of *E. faecium* isolates included in the study, the selected unitigs for node features, two adjancency matrices derived from SNPS and FCGR features. Here is the structure of the `data` folder

```
./data/
├── ast_labels.csv        # Susceptibility testing data
├── config.yaml           # Configuration file
├── extracted_unitigs     # Directory containing extracted unitigs for each isolate (Node features)
├── fcgr_adj_matrix.csv   # FCGR-based adjacency matrix (Edge construction)
├── snps_adj_matrix.csv   # SNP-based adjacency matrix (Edge construction)
├── whole.ids             # All isolate IDs included in the analysis
├── train.ids             # Training isolate IDs (for training step)
├── val.ids               # Validation isolate IDs (for training step)
└── predict.ids           # Prediction isolate IDs (for prediction step)
```

Here, we used [Hydra](https://hydra.cc) to manage configuration settings. For the full list of configuration settings, please see [here](./conf/config.yaml)

### Configuration table
| Argument                               | Description | 
|----------------------------------------|-------------|
| data.input_dir  | Directory containing node feature (unitigs) files |
| data.labels  | CSV file of susceptibility data (0: susceptible/1: resistant) |
| data.antimicrobial  | Column name of tested antimicrobial |
| data.whole_ids  | All isolate IDs included in the analysis |
| data.train_ids  | Training isolate IDs |
| data.val_ids  | Validation isolate IDs |
| data.adj_matrix.file_path_1  | The first adjacency matrix file |
| data.adj_matrix.file_path_2  | The second adjacency matrix file |
| data.model_checkpoint.dirpath  | Output directory to save the model checkpoint |
| trainer.seed  | Set random state for the training |
| trainer.max_epochs  | Number of training epochs (Default: 200) |
| trainer.accelerator  | Training device (Default: "auto") |
| trainer.devices  | Number of training devices (Default: "auto") |
| trainer.model_checkpoint.monitor  | Quantity to monitor (Default: "val_loss") |
| trainer.model_checkpoint.mode  | One of {min, max} (Default: "min") |

### Run with `uv`
```
uv run src/train.py data.input_dir=data/extracted_unitigs \
                    data.antimicrobial=vancomycin \
                    data.labels=data/ast_labels.csv \
                    data.whole_ids=data/whole.ids \
                    data.train_ids=data/train.ids \
                    data.val_ids=data/val.ids \
                    adj_matrix.file_path_1=data/fcgr_adj_matrix.csv \
                    adj_matrix.file_path_2=data/snps_adj_matrix.csv \
                    trainer.model_checkpoint.dirpath=experiments/checkpoints 
```

### Train with `docker`
```
docker run --rm \
           -v ${PWD}/data:/amrgnn/data \
           -v ${PWD}/experiments:/amrgnn/experiments \
           amrgnn:latest \
           src/train.py \
           data.input_dir=data/extracted_unitigs \
           data.antimicrobial=vancomycin \
           data.labels=data/ast_labels.csv \
           data.whole_ids=data/whole.ids \
           data.train_ids=data/train.ids \
           data.val_ids=data/val.ids \
           adj_matrix.file_path_1=data/fcgr_adj_matrix.csv \
           adj_matrix.file_path_2=data/snps_adj_matrix.csv \
           trainer.model_checkpoint.dirpath=experiments/checkpoints 
```

### Train with `apptainer`
```
apptainer exec --nv \
               --pwd /amrgnn \
               --bind ${PWD}/data:/amrgnn/data \
               --bind ${PWD}/experiments:/amrgnn/experiments \
               amrgnn.sif \
               uv run src/train.py\
               data.input_dir=data/extracted_unitigs \
               data.antimicrobial=vancomycin \
               data.labels=data/ast_labels.csv \
               data.whole_ids=data/whole.ids \
               data.train_ids=data/train.ids \
               data.val_ids=data/val.ids \
               adj_matrix.file_path_1=data/fcgr_adj_matrix.csv \
               adj_matrix.file_path_2=data/snps_adj_matrix.csv \
               trainer.model_checkpoint.dirpath=experiments/checkpoints 
```

## Predict AMR phenotype

### Configuration table
| Argument                               | Description | 
|----------------------------------------|-------------|
| data.input_dir  | Directory containing node feature (unitigs) files |
| data.labels  | CSV file of susceptibility data (0: susceptible/1: resistant) |
| data.antimicrobial  | Column name of tested antimicrobial |
| data.whole_ids  | All isolate IDs included in the analysis |
| data.predict_ids  | Isolate IDs for prediction |
| data.adj_matrix.file_path_1  | The first adjacency matrix file |
| data.adj_matrix.file_path_2  | The second adjacency matrix file |
| data.model_checkpoint.dirpath  | Directory containing the model checkpoint |
| prediction.outdir  | Output directory for prediction results |

### Predict vancomycin resistance with `uv`
```
uv run src/predict.py data.input_dir=data/extracted_unitigs \
                    data.antimicrobial=vancomycin \
                    data.labels=data/ast_labels.csv \
                    data.whole_ids=data/whole.ids \
                    data.predict_ids=data/predict.ids \
                    adj_matrix.file_path_1=data/fcgr_adj_matrix.csv \
                    adj_matrix.file_path_2=data/snps_adj_matrix.csv \
                    trainer.model_checkpoint.dirpath=experiments/checkpoints \
                    prediction.outdir=experiments/results
```

### Predict with `docker`
```
docker run --rm \
           -v ${PWD}/data:/amrgnn/data \
           -v ${PWD}/experiments:/amrgnn/experiments \
           amrgnn:latest \
           src/predict.py \
           data.input_dir=data/extracted_unitigs \
           data.antimicrobial=vancomycin \
           data.labels=data/ast_labels.csv \
           data.whole_ids=data/whole.ids \
           data.predict_ids=data/predict.ids \
           adj_matrix.file_path_1=data/fcgr_adj_matrix.csv \
           adj_matrix.file_path_2=data/snps_adj_matrix.csv \
           trainer.model_checkpoint.dirpath=experiments/checkpoints \
           prediction.outdir=experiments/results
```

### Predict with `apptainer`
```
apptainer exec --nv \
               --pwd /amrgnn \
               --bind ${PWD}/data:/amrgnn/data \
               --bind ${PWD}/experiments:/amrgnn/experiments \
               amrgnn.sif \
               src/predict.py \
               data.input_dir=data/extracted_unitigs \
               data.antimicrobial=vancomycin \
               data.labels=data/ast_labels.csv \
               data.whole_ids=data/whole.ids \
               data.predict_ids=data/predict.ids \
               adj_matrix.file_path_1=data/fcgr_adj_matrix.csv \
               adj_matrix.file_path_2=data/snps_adj_matrix.csv \
               trainer.model_checkpoint.dirpath=experiments/checkpoints \
               prediction.outdir=experiments/results
```

The output files (`prediction_results.csv`) is a csv file including 4 columns

| Column                               | Description | 
|----------------------------------------|-------------|
| isolate_ids  | Isolate id |
| y_true | True AST phenotype (Resistant/Susceptible) |
| y_proba | Predicted probability from AMR-GNN |
| y_pred | Predicted AST phenotype (Resistant/Susceptible) |

## Model interpretation

We used Integrated Gradients for feature attribution, utilizing the reference genome's unitig features as the baseline.

### Run with `uv`
```
uv run src/explain.py data.input_dir=data/extracted_unitigs \
                      data.antimicrobial=vancomycin \
                      data.labels=data/ast_labels.csv \
                      data.whole_ids=data/whole.ids \
                      adj_matrix.file_path_1=data/fcgr_adj_matrix.csv \
                      adj_matrix.file_path_2=data/snps_adj_matrix.csv \
                      trainer.model_checkpoint.dirpath=experiments/checkpoints \
                      explainer.out_fp="experiments/explain/IG_attribution.csv"
```

### Run with `docker`
```
docker run --rm \
           -v ${PWD}/data:/amrgnn/data \
           -v ${PWD}/experiments:/amrgnn/experiments \
           amrgnn:latest \
           src/explain.py \
           data.input_dir=data/extracted_unitigs \
           data.antimicrobial=vancomycin \
           data.labels=data/ast_labels.csv \
           data.whole_ids=data/whole.ids \
           adj_matrix.file_path_1=data/fcgr_adj_matrix.csv \
           adj_matrix.file_path_2=data/snps_adj_matrix.csv \
           trainer.model_checkpoint.dirpath=experiments/checkpoints \
           explainer.out_fp="experiments/explain/IG_attribution.csv"
```

### Run with `apptainer`
```
apptainer exec --nv \
               --pwd /amrgnn \
               --bind ${PWD}/data:/amrgnn/data \
               --bind ${PWD}/experiments:/amrgnn/experiments \
               amrgnn.sif \
               src/explain.py \
               data.input_dir=data/extracted_unitigs \
               data.antimicrobial=vancomycin \
               data.labels=data/ast_labels.csv \
               data.whole_ids=data/whole.ids \
               data.predict_ids=data/predict.ids \
               adj_matrix.file_path_1=data/fcgr_adj_matrix.csv \
               adj_matrix.file_path_2=data/snps_adj_matrix.csv \
               trainer.model_checkpoint.dirpath=experiments/checkpoints \
               explainer.out_fp="experiments/explain/IG_attribution.csv"
```

The output consists of a CSV file comprising a matrix of IG scores, where rows represent isolates and columns represent features.


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