# AMR-GNN: A multi-representation graph neural network framework to enable genomic antimicrobial resistance prediction

This is the code repository of the [paper](https://www.biorxiv.org/)

## Abstract
Whole-genome sequencing (WGS) data are an invaluable resource for understanding antimicrobial resistance (AMR) mechanisms. However, WGS data are high-dimensional and the lack of standardized genomic representations is a key barrier to AMR prediction. To fully explore these high-resolution data, we propose AMR-GNN, a graph deep learning-based framework that integrates multiple genomic representations with graph neural networks (GNN) to enable AMR prediction from genomic sequence data. We tested AMR-GNN with <em>Pseudomonas aeruginosa</em>, a clinically relevant Gram-negative bacterial pathogen known for its complex AMR mechanisms. We demonstrate that AMR-GNN addresses several key problems in AMR prediction with data-driven machine learning (ML) approaches, including using multiple genomic representations to enhance performance, mitigate the influence of clonal relationships, and identify informative biomarkers to provide explainability and generate novel hypotheses. Follow-up validation on the largest publicly available dataset spanning both Gram-negative and Gram-positive species highlights AMR-GNN’s broad applicability in detecting AMR in diverse and clinically relevant pathogen-drug combinations.

## Overview of AMR-GNN Workflow
![plot](./assets/workflow.png)

## System requirements and Installation

Create conda environment

```
conda env create -f envs/amr_gnn.yml
```

## AMR prediction
We use Hydra to manage configuration settings, all of which are specified in the [config.yaml](./conf/config.yaml) file. If you are not familiar with Hydra, please refer to their [documentation](https://hydra.cc/docs/intro/).

To run AMR-GNN, simply execute the following command:
```
python src/amr_gnn.py 
```

The output includes the best model checkpoint (if `del_ckpt` is set to `False`) along with the predicted results for the test set (`test_results.csv`).

To repeat the experiment with multiple random splits, using precomputed adjacency matrices can reduce overhead time:

```
python utils/precompute_adjacency_matrix.py ${DIST_FP} ${ANTIMICROBIAL} ${LABEL_FP} ${OUTDIR}
```

### Demo
We provide real-world data (PATRIC database) for predicting vancomycin and linezolid resistance in <em>Enterococcus faecium</em> (Figure 4c,g). Please download the [dataset](https://drive.google.com/file/d/1Ub0ngfWhoKsXUNf-ToyyYcRlWzxRMYfX/view?usp=sharing), extract it, and place it in the main directory. Then, execute the following commands:

```
cd experiments/efaecium
conda activate amr_gnn

for drug in vancomycin linezolid; do
  for run_id in {0..9}; do
    python ../../src/amr_gnn.py \
      -cp "${PWD}" \
      data.run_id="${run_id}" \
      data.antimicrobial="${drug}"
  done
done
```

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


