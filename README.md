# Demography Optimization via Neural Network Inference

# Introduction
Diffusion Approximation of Demographic Inference ([dadi](https://dadi.readthedocs.io/en/latest/)) is a powerful software tool for simulating the joint frequency spectrum (FS) of genetic variation among multiple populations and employing the FS for population-genetic inference. Here we introduce donni, a supervised machine learning-based framework for easier application of dadi's underlying demographic models. These machine learning models were trained on dadi-simulated data and can be used to make quick predictions on dadi demographic model parameters given FS input data from user and specified demographic model. The pipeline we used to train the machine learning models are also available here for users interested in using the same framework to train a new predictor for their customized demographic models.

# Installation
## Get the donni repo
Clone this repo to your local directory and `cd` into the `donni` dir
```console
$ git clone https://github.com/lntran26/donni.git
$ cd donni/
```

## Set up your python environment and install the donni pipeline
We recommend you start by creating a new `conda` environment. This can be done using the command below, which will create a new `conda` env called `donni` and installed the required packages to this env. The env can then be activated for each subsequent use.

```console
$ conda env create -f environment.yml
$ conda activate donni
```

# User manual

After installation, users can check for successful installation or get help information using:
```console
$ donni -h
usage: donni [-h] {generate_data,train,infer,plot} ...

Demography Optimization via Neural Network Inference

positional arguments:
  {generate_data,train,infer,plot}
    generate_data       Simulate allele frequency data from demographic history models
    train               Train MLPR with simulated allele frequency data
    infer               Infer demographic history parameters from allele frequency with trained MLPRs
    plot                Plot trained MLPRs inference accuracy and CI coverage

optional arguments:
  -h, --help            show this help message and exit
```

There are four subcommands in `donni` and the detailed usage for each subcommand can be found below:
- [`generate_data`](#generating-simulated-afs)
- [`train`](#hyperparameter-tuning-and-training-the-MLPR)
- [`infer`](#inferring-demographic-history-parameters-from-allele-frequency-data)
- [`plot`](#plotting-trained-MLPRs-accuracy-and-confidence-interval-coverage)

To display help information for each subcommand, users can use `-h`. For example:
```console
$ donni generate_data -h
```

# Inferring demographic history from allele frequency data
## Specifying a model and input data
To infer demographic model parameters with donni, users need to specify the desired demographic model name and the path to allele frequency data. 

```console
$ donni infer --input_fs examples/data/1d_ns20_sfs.fs --model three_epoch
```
This command will automatically download the relevant trained MLPRs used for inference to the user's home directory and the download location will be printed to the command line. The file size of one trained MLPR varies from a few Kb to ~100 Mb depending on the sample size of the input data and the number of populations in the demographic model. The number of trained MLPRs downloaded will correspond to the number of parameters in the requested demographic model (i.e. one trained MLPR per parameter.)

```console
Downloading model: /iplant/home/shared/donni/three_epoch/unfolded/ss_20/v0.0.1/tuned_models/param_01_predictor to /Users/username/Library/Caches/donni/0.0.1/three_epoch_unfolded_ns_20
Downloading model: /iplant/home/shared/donni/three_epoch/unfolded/ss_20/v0.0.1/tuned_models/param_02_predictor to /Users/username/Library/Caches/donni/0.0.1/three_epoch_unfolded_ns_20
Downloading model: /iplant/home/shared/donni/three_epoch/unfolded/ss_20/v0.0.1/tuned_models/param_03_predictor to /Users/username/Library/Caches/donni/0.0.1/three_epoch_unfolded_ns_20
Downloading model: /iplant/home/shared/donni/three_epoch/unfolded/ss_20/v0.0.1/tuned_models/param_04_predictor to /Users/username/Library/Caches/donni/0.0.1/three_epoch_unfolded_ns_20
Downloading model: /iplant/home/shared/donni/three_epoch/unfolded/ss_20/v0.0.1/tuned_models/param_05_predictor to /Users/username/Library/Caches/donni/0.0.1/three_epoch_unfolded_ns_20
Downloading QC: /iplant/home/shared/donni/three_epoch/unfolded/ss_20/v0.0.1/plots/theta_1000_coverage.png to /Users/username/Library/Caches/donni/0.0.1/three_epoch_unfolded_ns_20_QC
Downloading QC: /iplant/home/shared/donni/three_epoch/unfolded/ss_20/v0.0.1/plots/theta_1000_param_01_accuracy.png to /Users/username/Library/Caches/donni/0.0.1/three_epoch_unfolded_ns_20_QC
Downloading QC: /iplant/home/shared/donni/three_epoch/unfolded/ss_20/v0.0.1/plots/theta_1000_param_02_accuracy.png to /Users/username/Library/Caches/donni/0.0.1/three_epoch_unfolded_ns_20_QC
Downloading QC: /iplant/home/shared/donni/three_epoch/unfolded/ss_20/v0.0.1/plots/theta_1000_param_03_accuracy.png to /Users/username/Library/Caches/donni/0.0.1/three_epoch_unfolded_ns_20_QC
Downloading QC: /iplant/home/shared/donni/three_epoch/unfolded/ss_20/v0.0.1/plots/theta_1000_param_04_accuracy.png to /Users/username/Library/Caches/donni/0.0.1/three_epoch_unfolded_ns_20_QC
Downloading QC: /iplant/home/shared/donni/three_epoch/unfolded/ss_20/v0.0.1/plots/theta_1000_param_05_accuracy.png to /Users/username/Library/Caches/donni/0.0.1/three_epoch_unfolded_ns_20_QC
Downloading QC: /iplant/home/shared/donni/three_epoch/unfolded/ss_20/v0.0.1/plots/theta_1000_QC.txt to /Users/username/Library/Caches/donni/0.0.1/three_epoch_unfolded_ns_20_QC

Files for the requested model and configuration have already been downloaded to the /Users/username/Library/Caches/donni/0.0.1/three_epoch_unfolded_ns_20 folder.
If you want to redownload delete the directory
Check /Users/username/Library/Caches/donni/0.0.1/three_epoch_unfolded_ns_20_QC for the quality of MLPR models
```

Once downloaded, donni will use the trained MLPRs to infer the demographic parameter values and confidence intervals (default: 95% CI) for the user's input allele frequency data.

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
## Supported models and sample sizes

donni currently supports all demographic models in the [dadi API](https://dadi.readthedocs.io/en/latest/api/dadi/) as well as the models from [Portik et al.](https://github.com/dportik/dadi_pipeline). The supported sample sizes are 10, 20, 40, 80, and 160 chromosomes per population (up to 40 chromosomes only for three-population models). Input allele frequency spectra with a different sample size will be automatically down-projected to the closest available supported size before inference. donni will also automatically detect whether the input data is a folded or unfolded spectra.

If the requested model is not available, users will see this message:
```console
$ donni infer --input_fs examples/data/1d_ns20_sfs.fs --model foo
Files for the requested model and configuration do not exist.
```
## Data availability
The trained MLPRs (along with accuracy score and confidence interval coverage plots) are publicly available on the University of Arizona CyVerse Data Store (https://de.cyverse.org/data/ds/iplant/home/shared/donni).


## Specifying custom confidence intervals
Users can specify the confidence intervals they want with the `--cis` argument. For example, the 80th and 90th percent confidence intervals can be requested with the following command:

```console
$ donni infer --input_fs examples/data/1d_ns20_sfs.fs --model three_epoch --cis 80 90
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

If users have [trained their own MLPRs](#training-custom-mlprs), they can direct donni to the trained MLPRs file directory with the `--mlpr_dir` argument.

```console
$ donni infer --input_fs examples/data/1d_ns20_sfs.fs --model two_epoch --mlpr_dir examples/data/two_epoch_20_mlprs/
```

```console
# nu	T	misid	theta	nu_lb_95	nu_ub_95	T_lb_95	T_ub_95	misid_lb_95	misid_ub_95
0.14757507548097556	1.1754731225311374	0.03136719895731105	6925.010938463936	0.006715137102134652	2.6762432828151432	-0.1807778278375065	2.1900599819927873	-0.03685973417197391	0.08007636692730616

# CIs:    |----------95----------|	
# nu:     [  0.006715,   2.676243]	
# T:      [ -0.180778,   2.190060]	
# misid:  [ -0.036860,   0.080076]
```

# Training custom MLPRs
The three subcommands `generate_data`, `train`, and `plot` are for training and testing trained MLPRs. This is the procedure we used to produce all the MLPRs in the current library. Users can use the same method outlined below to create their custom demographic model with dadi and produce the corresponding trained MLPRs.

## Generating simulated AFS

### Specifying a custom demographic history model
donni uses dadi to simulate allele frequency data for training the MLPR. Hence, the custom model must be a Python script (`.py` file) in the format that dadi uses (see https://dadi.readthedocs.io/en/latest/user-guide/specifying-a-model/). The path to this file should be passed into donni with `--model_file` and the specific model in the file with `--model`. We can start building our command:

```console
$ donni generate_data --model out_of_africa --model_file donni/custom_models.py
```
### Specifying the rest of the arguments
Three additional arguments are required for generating the training data: `--n_samples`, `--sample_sizes`, and `--outfile`. 

The `--sample_sizes` argument defines the number of sampled chromosomes per population hence the size of the simulated AFS. A three-population model should have three numbers separated by a space for the number of chromosomes in each population. For example:
```console
--sample_sizes 10 10 10 
```
The `--n_samples` argument specifies the number of AFS with unique parameter labels to be generated. We used 5000 AFS for the training set and 1000 AFS for test set. The optional argument `--seed` can be used for reproducibility as well as ensuring different AFS being generated for training vs. testing.
```console
--n_samples 5000 --seed 1
```
The `--outfile` argument specifies the file name and path to save the output data. This is a pickled Python dictionary with parameter labels as keys and simulated AFS as values.
```console
--outfile data/train_5000
```
### Other optional arguments:

The argument `--save_individual_fs` can be used to save each simulated AFS as a single file instead of all in one dictionary. The parameter labels will be saved separately in a pickled file `true_log_params`. This usage required specifying an additional argument `--outdir`, which specifies the directory where all the individual AFS files and the `true_log_params` file will be saved to. Note that for this usage, `--outfile` still needs to be specified, as it is also used to save the corresponding QC files (for checking simulated AFS quality).
```console
--save_individual_fs --outdir test_fs --outfile test_fs/test_100_theta_1000
```

The `--grids` argument is used by the [dadi](https://dadi.readthedocs.io/en/latest/user-guide/simulation-and-fitting/#grid-sizes-and-extrapolation) simulation engine to calculate the AFS. donni will calculate the appropriate grids by default based on the specified `--sample_sizes`. Higher grid points can improve the quality of the simulated AFS. donni will automatically check the quality of the spectra generated, which can be turned off using the `--no_fs_qual_check` option. 

The `--theta` argument is used to control the variance (noise) in the simulated AFS by scaling the spectra with the theta value passed in then resampling. By default, `--theta` is 1 (no scaling, no sampling), which is used for generating the training AFS (no noise). For generating test AFS, we often use `--theta 1000` to simulate moderately noisy AFS. If the value passed into `--theta` is > 1, the AFS generated will be scaled by the passed in value. Because we often only want to do this to generate noisy AFS for testing, by default donni will also Poisson-sample from the scaled AFS. If this is not desirable, use `--no_sampling` argument for scaling without sampling. Similarly, all simulated AFS are, by default, normalized before being saved. The `--non_normalize` argument allows bypassing this and will generate non-normalized AFS. 

Users can use the `--folded` argument if they want to generate folded AFS. By default, unfolded AFS will be generated.
​

donni can also generate bootstraped AFS data with the `--bootstrap` argument. For this usage, the `--n_bstr` argument is required to specify how many bootstraped AFS to generate per simulated AFS. 

By default, donni will use all available CPUs to simulate the AFS in parallel. Users can control the number of CPUs used with `--n_cpu`.

## Generating hyperparmeters for tuning:
To prepare the next step (tuning), we also generate a dictionary file containing the hyperparmeters for the MLPRs that will be used during automatic hyperparemeter tuning. Since the number of hidden layers and neurons in the MLPR depend on `--sample_sizes`, we also generate the appropriate hyperparmeters in this step. `donni generate_data` has three arguments for this purpose: `--generate_tune_hyperparam`, `--generate_tune_hyperparam_only` (no AFS simulation), and `--hyperparam_outfile`. An example command for generating the hyperparmeters dictionary would be:
```console
--generate_tune_hyperparam --hyperparam_outfile data/param_dict_tune
```

## Generating data: full example commands
To generate 5000 training AFS for the out_of_africa model found in the donni/donni/custom_models.py file and the hyperparmeters:

```console
$ donni generate_data --model out_of_africa --model_file donni/custom_models.py --n_samples 5000 \
--sample_sizes 10 10 10 --seed 1 --outfile data/train_5000 --generate_tune_hyperparam \
--hyperparam_outfile data/param_dict_tune
```
To generate 1000 test AFS with moderate noise for the same model:

```console
$ donni generate_data --model out_of_africa --model_file donni/custom_models.py --n_samples 1000 \
--sample_sizes 10 10 10 --seed 100 --theta 1000 --outfile data/test_1000_theta_1000
```
To generate only the hyperparmeters without simulating any AFS
```console
$ donni generate_data --model split_mig --n_samples 1 --sample_sizes 40 40 --outfile foo \
--generate_tune_hyperparam_only --hyperparam_outfile data/param_dict_tune
```

To generate 100 test AFS (non-normalized) and save them individually (used in optimization with dadi-cli)
```console
$ donni generate_data --model split_mig --n_samples 100 --sample_sizes 160 160 \
--theta 1000 --seed 100 --non_normalize --save_individual_fs \
--outdir test_fs --outfile test_fs/test_100_theta_1000"
```

## Hyperparameter tuning and training the MLPR
​

## Plotting trained MLPRs accuracy and confidence interval coverage


# Requirements
1. Python 3.9+
2. [dadi](https://dadi.readthedocs.io/en/latest/)
3. [scikit-learn](https://scikit-learn.org/1.0/) 1.2.0
4. [MAPIE](https://mapie.readthedocs.io/en/latest/) 0.6.1


# References
1. [Gutenkunst et al., *PLoS Genet*, 2009.](https://journals.plos.org/plosgenetics/article?id=10.1371/journal.pgen.1000695)
