# Neural Network Runtime Monitoring
A library to benchmark neural network runtime monitoring approaches on popular computer vision datasets 

## Install

The installation proposed here is based on conda.

*To install conda on your machine you can follow [this site](https://doc.ubuntu-fr.org/miniconda)*.
 
Then, you simply need to:
* Clone this git repository
* Install all dependencies in a conda virtual environment

        $ conda env create -f environment.yml
* Activate the environment

        $ conda activate neural-network-monitoring-benchmark

## Overview

This benchmarking library is built around four base classes:

<details>
<summary><b>Dataset</b></summary>
This class allows to load a specific dataset. 

A dataset is configured by specifying 
* the **name** of the dataset, 
* the **split** (train/test),
* the **network** that will be used to process it, 
* the **additional transform**  applied to images (e.g., brightness changes), 
* the **adversarial attack** type applied to images, 
* and the **batch size**.

The valid configuration parameters are defined in *Params/params_dataset.py*.

See *dataset.py* for a more detailed documentation.
</details>

<details>
<summary><b>FeatureExtractor</b></summary>
The FeatureExtractor class plays a pivotal role in this benchmarking library, responsible for efficiently extracting various essential components from the dataset. It enables the extraction of:
 * Data Features
 * Logits
 * Softmax values
 * Predictions
 * Labels
</details> 

<details>
<summary><b>Monitor</b></summary>
The Monitor class represents different monitoring approaches. It encompasses various monitoring methods with different functionalities.
</details>

<details>
<summary><b>Evaluator</b></summary>
The Evaluator class calculates performance metrics to assess the effectiveness of monitoring methods. It computes key metrics like AUROC (Area Under the Receiver Operating Characteristic curve), AUPR (Area Under the Precision-Recall curve), and TNR95TPR (True Negative Rate at 95% True Positive Rate).

</details>
<summary><b>logit transform comparison</b></summary>

In this section, we introduce the monitors_logits library. It leverages logit-based approaches to monitor neural network predictions. We have unified the scores to ensure that higher scores indicate rejection.

* Optimal Hyperparameter Tuning: We use the Nemenyi test to find the optimal hyperparameter values for each monitor.
* Monitor Comparison: Wilcoxon tests are employed to compare the performance of different monitors, with further analysis using the Nemenyi test for multiple comparisons.



### References
If you found this library useful, please consider citing the following works:

*To appear soon*

### Acknowledgements
*This research has benefited from the AI Interdisciplinary Institute ANITI. 
ANITI is funded by the French "Investing for the Future – PIA3" program
under the Grant agreement No ANR-19-PI3A-0004.*