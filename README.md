# Demography Optimization via Neural Network Inference

## Introduction
Diffusion Approximation of Demographic Inference ([dadi](https://dadi.readthedocs.io/en/latest/)) is a powerful software tool for simulating the joint frequency spectrum (FS) of genetic variation among multiple populations and employing the FS for population-genetic inference. Here we introduce donni, a supervised machine learning-based framework for easier application of dadi's underlying demographic models. These machine learning models were trained on dadi-simulated data and can be used to make quick predictions on dadi demographic model parameters given FS input data from user and specified demographic model. The pipeline we used to train the machine learning models are also available here for users interested in using the same framework to train a new predictor for their customized demographic models.

## Installation
### Get the donni repo
Clone this repo to your local directory and `cd` into the `donni` dir
```console
$ git clone https://github.com/lntran26/donni.git
$ cd donni/
```

### Set up your python environment and install the donni pipeline
We recommend you start by creating a new `conda` environment. This can be done using the command below, which will create a new `conda` env called `donni` and installed the required packages to this env. The env can then be activated for each subsequent use.

```console
$ conda env create -f environment.yml
$ conda activate donni
```

## Requirements
1. Python 3.9+
2. [dadi](https://dadi.readthedocs.io/en/latest/)
3. [scikit-learn](https://scikit-learn.org/1.0/) 1.2.0
4. [MAPIE](https://mapie.readthedocs.io/en/latest/) 0.6.1


## References
1. [Gutenkunst et al., *PLoS Genet*, 2009.](https://journals.plos.org/plosgenetics/article?id=10.1371/journal.pgen.1000695)
