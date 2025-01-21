This repository serves as official source code associated with the paper "Can we Defend Against the Unknown? An Empirical Study About Threshold Selection for Neural Network Monitoring" (authors: Khoi Tran Dang, Kevin Delmas, Jérémie Guiochet, Joris Guérin). 

The paper is accepted for the 40th Conference on Uncertainty in Artificial Intelligence (UAI 2024). Its preprint is also available on [Arxiv](https://arxiv.org/abs/2405.08654)/[HAL](https://hal.science/hal-04579393/document).

It is highly recommended to take a look at the paper first, the code can be used for reproducibility.

# Reproducing Experiments

## 1. Preparations 

Clone the main repository (https://github.com/jorisguerin/neural-network-monitoring-benchmark) then change directory into the folder related to the paper `/script_uai/`. 

### 1.1. Environment 

The installation proposed here is based on conda. The installation of conda is proposed on its official website.

If you have conda installed, you simply need to:
* Install all dependencies in a conda virtual environment

        $ conda env create -f env.yml
* Activate the environment

        $ conda activate thresholding_runtime_monitor

### 1.2. Datasets, networks, monitors, evaluators

Please refer to the main README.md in the root folder of the github for more information on these preparatory codes. 

Datasets like CIFAR10, CIFAR100, SVHN, LSUN, and TinyImageNet are used, with various transformations applied to perturb the original images (e.g., brightness changes and adversarial attacks). Refer to Section 4.1 of the paper for more details.

See *dataset.py* for a more detailed documentation.

Four distinct monitoring techniques are employed: Mahalanobis and Outside-the-Box (OtB) for feature-based analysis, and Max Softmax Probability and Energy for logit-based methods. Additional techniques are also available in Joris Guérin's GitHub repository.

See *monitors.py* for a more detailed documentation.

## 2. Paper code 

For quick access, the raw CSV results for all experimental settings are located inside the folder `script_uai/result_csv/`. There are two results csv files:
- `Result-Thresh-Opt_resampling.csv`
- `Result-Thresh-Opt.csv`.

The former includes results where the effectiveness measures incorporate F1+oversampling and g-mean, while the latter includes results where the effectiveness measures involve F1 without oversampling and g-mean. These raw CSV files encompasses the evaluation metric scores for every combination and approach presented in the experimental design. It is advisable to take a look at this file after reading the paper to gain a comprehensive overview of the experiments. 

In-depth analysis based on these CSV files, including statistical tests and other crucial information, is available in the Jupyter notebook named `Thresholding_Exp_Notebook.ipynb`, located in the folder `script_uai/`.

To conduct experiments proposed in the paper, you simply need to open the jupyter notebook named `Thresholding_Exp_Notebook.ipynb` and run cells inside. 

You can reproduce all results analysis by running the main notebook. If you want to reconduct the experiments, you can rename or delete the already-existed results csv and rerun the notebook. 

Some python scripts in the folder: 
- `oversampling.py`: this defines a simple function to oversample minor class within an imbalanced set.
- `thresh_strategies_dataset.py`: this defines the main python class to construct threshold optimization sets and threshold evaluation sets for experiments. 
- `wilcoxon_signed_rank.py`: this defines wilcoxon signed rank test and visualization functions for it. 
- `friedman_nemenyi.py`: this defines fried man nemenyi test and visualization functions for it. 
- `Supplementary Analysis.ipynb`: this file is some extra stat analysis that we have done, not put into the paper.

For more information of the experiments design and the objectives, please refer to the paper associated. 



