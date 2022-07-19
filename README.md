# Machine Learning Applications for Diffusion Approximation of Demographic Inference

## Introduction
Diffusion Approximation of Demographic Inference ([dadi](https://dadi.readthedocs.io/en/latest/)) is a powerful software tool for simulating the joint frequency spectrum (FS) of genetic variation among multiple populations and employing the FS for population-genetic inference. Here we introduce machine learning-based tools for easier application of dadi's underlying demographic models. These machine learning models were trained on dadi-simulated data and can be used to make quick predictions on dadi demographic model parameters given FS input data from user and specified demographic model. The pipeline we used to train the machine learning models are also available here for users interested in using the same framework to train a new predictor for their customized demographic models.

## Installation
### Get the dadi-ml repo
Clone this repo to your local directory and `cd` into the `dadi-ml` dir
```console
$ git clone https://github.com/lntran26/dadi-ml.git
$ cd dadi-ml/
```

### Set up your python environment to run the dadi-ml pipeline
We recommend you start by creating a new `conda` environment. This can be done using the command below, which will
create a new `conda` env called `dadi-ml`.

```console
$ conda env create -f environment.yml
$ conda activate dadi-ml
```

## Requirements
1. Python 3.9+
2. [dadi](https://dadi.readthedocs.io/en/latest/)
3. [scikit-learn](https://scikit-learn.org/1.0/) 1.0.2
4. [MAPIE](https://mapie.readthedocs.io/en/latest/) 0.3.1


## References
1. [Gutenkunst et al., *PLoS Genet*, 2009.](https://journals.plos.org/plosgenetics/article?id=10.1371/journal.pgen.1000695)
