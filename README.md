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

</details> 

<details>
<summary><b>Monitor</b></summary>

</details>

<details>
<summary><b>Evaluator</b></summary>

</details>


### References
If you found this library useful, please consider citing the following works:

*To appear soon*

### Acknowledgements
*This research has benefited from the AI Interdisciplinary Institute ANITI. 
ANITI is funded by the French "Investing for the Future – PIA3" program
under the Grant agreement No ANR-19-PI3A-0004.*