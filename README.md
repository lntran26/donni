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

## User manual

### Infer

donni has trained multilayer preceptron (MLP) for all of the demographic models in dadi as well as the models from Portik el al (CITE) stored on the University of Arizona CyVerse Data Store (https://cyverse.org/data-store) available for users. `donni infer` will by default try to download the trained MLP for the requested `--model` based on `--input_fs`.

```console
donni infer --input_fs examples/data/1d_ns20_sfs.fs --model three_epoch
```

The location will be printed to the command line:

```console
Downloading: /iplant/home/shared/donni/three_epoch/unfolded/ss_20/v0.0.1/tuned_models/param_01_predictor to /Users/username/Library/Caches/donni/0.0.1/three_epoch_unfolded_ns_20
Downloading: /iplant/home/shared/donni/three_epoch/unfolded/ss_20/v0.0.1/tuned_models/param_02_predictor to /Users/username/Library/Caches/donni/0.0.1/three_epoch_unfolded_ns_20
Downloading: /iplant/home/shared/donni/three_epoch/unfolded/ss_20/v0.0.1/tuned_models/param_03_predictor to /Users/username/Library/Caches/donni/0.0.1/three_epoch_unfolded_ns_20
Downloading: /iplant/home/shared/donni/three_epoch/unfolded/ss_20/v0.0.1/tuned_models/param_04_predictor to /Users/username/Library/Caches/donni/0.0.1/three_epoch_unfolded_ns_20
Downloading: /iplant/home/shared/donni/three_epoch/unfolded/ss_20/v0.0.1/tuned_models/param_05_predictor to /Users/username/Library/Caches/donni/0.0.1/three_epoch_unfolded_ns_20
```

Once downloaded, donni will attempt to infer the demographic parameter values and 95% confidence intervals for the user's data.

```console
# nuB	nuF	TB	TF	misid	theta	nuB_lb_95	nuB_ub_95	nuF_lb_95	nuF_ub_95	TB_lb_95	TB_ub_95	TF_lb_95	TF_ub_95	misid_lb_95	misid_ub_95
1.4589417590202813	0.27766527063507107	0.3348057792116058	0.678676374216568	0.016573499238520806	3078.400966873281	0.0026777917267467415	132.55568205629365	0.0007125091343149004	8.512893988606479	-0.5897319305793076	1.251661724536784	-0.1420600082022596	1.5295973418174724	-0.007558984976014427	0.08056165858500447

# CIs:    |----------95----------|	
# nuB:    [  0.002678, 132.555682]	
# nuF:    [  0.000713,   8.512894]	
# TB:     [ -0.589732,   1.251662]	
# TF:     [ -0.142060,   1.529597]	
# misid:  [ -0.007559,   0.080562]
```

Users can specify the confidence intervals they want with the `--cis` argument.

For example, if the user requests the 80th and 90th percent confidence intervals:

```console
donni infer --input_fs examples/data/1d_ns20_sfs.fs --model three_epoch --cis 80 90
```

```console
# nuB	nuF	TB	TF	misid	theta	nuB_lb_80	nuB_ub_80	nuF_lb_80	nuF_ub_80	TB_lb_80	TB_ub_80	TF_lb_80	TF_ub_80	misid_lb_80	misid_ub_80	nuB_lb_90	nuB_ub_90	nuF_lb_90	nuF_ub_90	TB_lb_90	TB_ub_90	TF_lb_90	TF_ub_90	misid_lb_90	misid_ub_90
1.4589417590202813	0.27766527063507107	0.3348057792116058	0.678676374216568	0.016573499238520806	3078.400966873281	0.02634476337211434	44.583151697085455	0.007657835658415717	1.457465130942518	-0.14865156750354747	0.7857211494174595	0.15328463274251425	1.3291811726081775	0.002226777650249062	0.052656001753191906	0.008203718136729264	91.29726937105971	0.002174982530860171	5.389966821614843	-0.299796473057088	0.9892394295337347	0.011915808230589464	1.4608120551998374	-0.0020380172128151852	0.06658196470780856

# CIs:    |----------80----------|	|----------90----------|	
# nuB:    [  0.026345,  44.583152]	[  0.008204,  91.297269]	
# nuF:    [  0.007658,   1.457465]	[  0.002175,   5.389967]	
# TB:     [ -0.148652,   0.785721]	[ -0.299796,   0.989239]	
# TF:     [  0.153285,   1.329181]	[  0.011916,   1.460812]	
# misid:  [  0.002227,   0.052656]	[ -0.002038,   0.066582]
```

If users have trained their own MLPs, they can direct donni to the directory with the `--mlpr_dir`.

```console
donni infer --input_fs examples/data/1d_ns20_sfs.fs --model two_epoch --mlpr_dir examples/data/two_epoch_20_mlprs/
```

```console
# nu	T	misid	theta	nu_lb_95	nu_ub_95	T_lb_95	T_ub_95	misid_lb_95	misid_ub_95
0.14757507548097556	1.1754731225311374	0.03136719895731105	6925.010938463936	0.006715137102134652	2.6762432828151432	-0.1807778278375065	2.1900599819927873	-0.03685973417197391	0.08007636692730616

# CIs:    |----------95----------|	
# nu:     [  0.006715,   2.676243]	
# T:      [ -0.180778,   2.190060]	
# misid:  [ -0.036860,   0.080076]
```


## Requirements
1. Python 3.9+
2. [dadi](https://dadi.readthedocs.io/en/latest/)
3. [scikit-learn](https://scikit-learn.org/1.0/) 1.2.0
4. [MAPIE](https://mapie.readthedocs.io/en/latest/) 0.6.1


## References
1. [Gutenkunst et al., *PLoS Genet*, 2009.](https://journals.plos.org/plosgenetics/article?id=10.1371/journal.pgen.1000695)
