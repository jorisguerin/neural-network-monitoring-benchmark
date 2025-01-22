# Neural Network Runtime Monitoring

A library to benchmark neural network runtime monitoring approaches on popular computer vision datasets. The focus is for now on image classification tasks and the aim is to extend the scope of the study to tackle object detection and image segmentation.

## Install

### Installation using Anaconda

The easiest installation process here is based on `conda`. *To install conda (Miniconda for a lighter package) on your machine, you can follow [this site](https://docs.anaconda.com/miniconda/) (for Windows users) or [this site](https://doc.ubuntu-fr.org/miniconda) (for Linux users)*.

The install then simply follows these steps:

* Clone this git reposotiry
* Install all dependencies in a conda virtual environment with 

        $ conda env create -f environment-2025.yml

* Activate the environment with

        conda activate nn-monitoring-benchmark

### Installation using pip

If you don't want to install conda, you can try and install the project using `pip`. Make sure you use the correct version of Python (`Python3.10`).

*A detailed installation process is yet to be written...*

## Overview

This benchmarking library is built around four base types of classes:

<details>
<summary><b>Datasets</b></summary>

The `Dataset` class allows to load a specific dataset and to perform additional transformations of the data, as applying perturbations (*covariate shifts*) or adversarial attacks. A dataset is configured by specifying:

* the **name** of the dataset, 
* the **split** (either *train* or *test*),
* the **network** that will be used to process it, 
* the **data perturbations** applied to images (e.g., brightness changes), 
* the **adversarial attack** type applied to images, 
* and the **batch size**.

The valid configuration parameters are defined in `Params/params_dataset.py`. For a more detailed documentation, see `dataset.py`.

</details>

<details>
<summary><b>Features Extractor</b></summary>

The `FeatureExtractor` class plays a pivotal role in this benchmarking library. It is responsible for efficiently extracting various essential components from the dataset and the model used to process it. It enables the extraction of key features to be used by diverse monitors:

* Data Features (latent representation of the data from some hidden layers of the models)
* Model logits values
* Model softmax values
* Model predictions
* Data labels

</details> 

<details>
<summary><b>Monitors</b></summary>

Implemented in the `Monitors/` directory, those classes represent different monitoring approaches. The monitors implemented are then used in the monitoring pipeline. A non-exhaustive list of the available monitors is presented below:

* `MSPMonitor` (*Max Softmax Probability monitor*)
* `MaxLogitsMonitor`
* `EnergyMonitor`
* `DOCTOR`
* `MahalanobisMonitor`
* `OTBMonitor` (*Outside-the-Box monitor*) 

</details>

<details>
<summary><b>Evaluator</b></summary>

The `Evaluator` class calculates performance metrics to assess the effectiveness of monitoring methods. It computes key metrics like AUROC (Area Under the Receiver Operating Characteristic curve), AUPR (Area Under the Precision-Recall curve), and TNR95TPR (True Negative Rate at 95% True Positive Rate).

</details>

### A demonstration notebook

This library comes with a `nb_demo.ipynb` jupyter notebook which allows you to see a complete pipeline for evaluating a monitor, for a given dataset and a given model, and compute performance metrics, usable for comparing the different monitoring strategies.

## Experiments using the library 

<details>
<summary><b>logit transform comparison</b></summary>

In this section, we introduce the monitors_logits library. It leverages logit-based approaches to monitor neural network predictions. We have unified the scores to ensure that higher scores indicate rejection.

* Optimal Hyperparameter Tuning: We use the Nemenyi test to find the optimal hyperparameter values for each monitor.
* Monitor Comparison: Wilcoxon tests are employed to compare the performance of different monitors, with further analysis using the Nemenyi test for multiple comparisons.

</details>

### References
If you found this library useful, please consider citing the following works:

*To appear soon*

### Acknowledgements
*This research has benefited from the AI Interdisciplinary Institute ANITI. 
ANITI is funded by the French "Investing for the Future – PIA3" program
under the Grant agreement No ANR-19-PI3A-0004.*