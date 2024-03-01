# Demography Optimization via Neural Network Inference

# Introduction
Diffusion Approximation of Demographic Inference ([dadi](https://dadi.readthedocs.io/en/latest/)) is a powerful software tool for simulating the joint frequency spectrum (FS) of genetic variation among multiple populations and employing the FS for population-genetic inference. Here we introduce donni, a supervised machine learning-based framework for easier application of dadi's underlying demographic models. These machine learning models were trained on dadi-simulated data and can be used to make quick predictions on dadi demographic model parameters given FS input data from user and specified demographic model. The pipeline we used to train the machine learning models are also available here for users interested in using the same framework to train a new predictor for their customized demographic models.

# Getting help
If you've found an apparent bug, please submit an Issue so we can address it. You can submit questions about usage to the dadi-user Google Group (http://groups.google.com/group/dadi-user).

# Installation
## Get the donni repo
Shallow clone this repo to your local directory and `cd` into the `donni` dir. We recommend using shallow clone for faster cloning.
```console
$ git clone https://github.com/lntran26/donni.git --depth 1
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
usage: donni [-h] {generate_data,train,infer,validate} ...

Demography Optimization via Neural Network Inference

positional arguments:
  {generate_data,train,infer,validate}
    generate_data       Simulate allele frequency data from demographic history models
    train               Train MLPR with simulated allele frequency data
    infer               Infer demographic history parameters from allele frequency with trained MLPRs
    validate            Validate trained MLPRs inference accuracy and CI coverage

optional arguments:
  -h, --help            show this help message and exit
```

There are four subcommands in `donni` and the detailed usage for each subcommand can be found below:
- [`generate_data`](#generating-simulated-afs)
- [`train`](#hyperparameter-tuning-and-training-the-MLPR)
- [`infer`](#inferring-demographic-history-parameters-from-allele-frequency-data)
- [`validate`](#validating-trained-MLPRs-accuracy-and-confidence-interval-coverage)

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
This command will automatically download the relevant trained MLPRs used for inference to the user's home directory and the download location will be printed to the command line. The file size of one trained MLPR varies from a few Kb to a few Mb, depending on the sample size of the input data and the number of populations in the demographic model. The number of trained MLPRs downloaded will correspond to the number of parameters in the requested demographic model (i.e. one trained MLPR per parameter.)

```console
Downloading model: /iplant/home/shared/donni/three_epoch/unfolded/ss_20/v0.9.0/tuned_models/param_01_predictor.keras to /home/lntran/.cache/donni/0.9.0/three_epoch_unfolded_ns_20
Downloading model: /iplant/home/shared/donni/three_epoch/unfolded/ss_20/v0.9.0/tuned_models/param_02_predictor.keras to /home/lntran/.cache/donni/0.9.0/three_epoch_unfolded_ns_20
Downloading model: /iplant/home/shared/donni/three_epoch/unfolded/ss_20/v0.9.0/tuned_models/param_03_predictor.keras to /home/lntran/.cache/donni/0.9.0/three_epoch_unfolded_ns_20
Downloading model: /iplant/home/shared/donni/three_epoch/unfolded/ss_20/v0.9.0/tuned_models/param_04_predictor.keras to /home/lntran/.cache/donni/0.9.0/three_epoch_unfolded_ns_20
Downloading model: /iplant/home/shared/donni/three_epoch/unfolded/ss_20/v0.9.0/tuned_models/param_05_predictor.keras to /home/lntran/.cache/donni/0.9.0/three_epoch_unfolded_ns_20
Downloading QC: /iplant/home/shared/donni/three_epoch/unfolded/ss_20/v0.9.0/plots/three_epoch_test_theta_1000_coverage_coverage.png to /home/lntran/.cache/donni/0.9.0/three_epoch_unfolded_ns_20_QC
Downloading QC: /iplant/home/shared/donni/three_epoch/unfolded/ss_20/v0.9.0/plots/three_epoch_test_theta_1000_param_01_accuracy_95_ci.png to /home/lntran/.cache/donni/0.9.0/three_epoch_unfolded_ns_20_QC
Downloading QC: /iplant/home/shared/donni/three_epoch/unfolded/ss_20/v0.9.0/plots/three_epoch_test_theta_1000_param_01_accuracy.png to /home/lntran/.cache/donni/0.9.0/three_epoch_unfolded_ns_20_QC
Downloading QC: /iplant/home/shared/donni/three_epoch/unfolded/ss_20/v0.9.0/plots/three_epoch_test_theta_1000_param_02_accuracy_95_ci.png to /home/lntran/.cache/donni/0.9.0/three_epoch_unfolded_ns_20_QC
Downloading QC: /iplant/home/shared/donni/three_epoch/unfolded/ss_20/v0.9.0/plots/three_epoch_test_theta_1000_param_02_accuracy.png to /home/lntran/.cache/donni/0.9.0/three_epoch_unfolded_ns_20_QC
Downloading QC: /iplant/home/shared/donni/three_epoch/unfolded/ss_20/v0.9.0/plots/three_epoch_test_theta_1000_param_03_accuracy_95_ci.png to /home/lntran/.cache/donni/0.9.0/three_epoch_unfolded_ns_20_QC
Downloading QC: /iplant/home/shared/donni/three_epoch/unfolded/ss_20/v0.9.0/plots/three_epoch_test_theta_1000_param_03_accuracy.png to /home/lntran/.cache/donni/0.9.0/three_epoch_unfolded_ns_20_QC
Downloading QC: /iplant/home/shared/donni/three_epoch/unfolded/ss_20/v0.9.0/plots/three_epoch_test_theta_1000_param_04_accuracy_95_ci.png to /home/lntran/.cache/donni/0.9.0/three_epoch_unfolded_ns_20_QC
Downloading QC: /iplant/home/shared/donni/three_epoch/unfolded/ss_20/v0.9.0/plots/three_epoch_test_theta_1000_param_04_accuracy.png to /home/lntran/.cache/donni/0.9.0/three_epoch_unfolded_ns_20_QC
Downloading QC: /iplant/home/shared/donni/three_epoch/unfolded/ss_20/v0.9.0/plots/three_epoch_test_theta_1000_param_05_accuracy_95_ci.png to /home/lntran/.cache/donni/0.9.0/three_epoch_unfolded_ns_20_QC
Downloading QC: /iplant/home/shared/donni/three_epoch/unfolded/ss_20/v0.9.0/plots/three_epoch_test_theta_1000_param_05_accuracy.png to /home/lntran/.cache/donni/0.9.0/three_epoch_unfolded_ns_20_QC
Downloading QC: /iplant/home/shared/donni/three_epoch/unfolded/ss_20/v0.9.0/plots/three_epoch_test_theta_1000_report.txt to /home/lntran/.cache/donni/0.9.0/three_epoch_unfolded_ns_20_QC

Finished downloading files to /home/lntran/.cache/donni/0.9.0/three_epoch_unfolded_ns_20 folder.
```

Once downloaded, donni will use the trained MLPRs to infer the demographic parameter values and confidence intervals (default: 95% CI) for the user's input allele frequency data.

```console
***Inferred demographic model parameters***
# nuB   nuF     TB      TF      misid   theta   nuB_lb_95       nuB_ub_95       nuF_lb_95       nuF_ub_95       TB_lb_95        TB_ub_95        TF_lb_95        TF_ub_95        misid_lb_95     misid_ub_95
0.4146571254600148      0.3869760036609383      0.5405870676040649      0.6910597085952759      0.01265975832939148     2489.12434337297        0.0022699636903605234   75.74593920815283       0.033626422287440526    4.45335593924669 -0.3420185887813568     1.4231927239894868      -0.13996889114379885    1.5220883083343506      -0.028875216841697693   0.05419473350048065

# CIs:    |----------95----------|
# nuB:    [  0.002270,  75.745939]
# nuF:    [  0.033626,   4.453356]
# TB:     [ -0.342019,   1.423193]
# TF:     [ -0.139969,   1.522088]
# misid:  [ -0.028875,   0.054195]

Check the plots in /home/lntran/.cache/donni/0.9.0/three_epoch_unfolded_ns_20_QC for accuracy scores of the downloaded model.
```

If users want to use `dadi` with the results from donni to do plotting or further analysis, they can export a file that can be read by [`dadi-cli`](https://dadi-cli.readthedocs.io/en/latest/), a command line interface for `dadi`. Users can use the `--export_dadi_cli` flag to store results that can be read by `dadi-cli` in a file with a user specified name which will have `.donni.pseudofit` as an extension.
For example the command

```console
$ donni infer --input_fs examples/data/1d_ns20_sfs.fs --model three_epoch --export_dadi_cli three_epoch_ns20
```

will generate a file `three_epoch_ns20.donni.pseudofit`, which can then be read by `dadi-cli` subcommands that use the `--demo-pop` and `--bestfit-p0-file` flags.

## Supported models and sample sizes

donni currently supports all demographic models in the [dadi API](https://dadi.readthedocs.io/en/latest/api/dadi/) as well as the models from [Portik et al.](https://github.com/dportik/dadi_pipeline). The supported sample sizes are 10, 20, 40, 80, and 160 chromosomes per population (up to 20 chromosomes only for three-population models). Input allele frequency spectra with a different sample size will be automatically down-projected to the closest available supported size before inference. donni will also automatically detect whether the input data is a folded or unfolded spectra.

If the requested model is not available, users will see this message:
```console
$ donni infer --input_fs examples/data/1d_ns20_sfs.fs --model foo
The requested demographic model does not exist on the CyVerse Data Store.
Users can check for available models at https://de.cyverse.org/data/ds/iplant/home/shared/donni
If the user has generated their own trained MLPRs, use --mlpr_dir
```
## Data availability
The trained MLPRs (along with accuracy score and confidence interval coverage plots) are publicly available on the University of Arizona CyVerse Data Store (https://de.cyverse.org/data/ds/iplant/home/shared/donni).


## Specifying custom confidence intervals
Users can specify the confidence intervals they want with the `--cis` argument. For example, the 80th and 90th percent confidence intervals can be requested with the following command:

```console
$ donni infer --input_fs examples/data/1d_ns20_sfs.fs --model three_epoch --cis 80 90
```

```console
***Inferred demographic model parameters***
# nuB   nuF     TB      TF      misid   theta   nuB_lb_80       nuB_ub_80       nuF_lb_80       nuF_ub_80       TB_lb_80        TB_ub_80        TF_lb_80        TF_ub_80        misid_lb_80     misid_ub_80     nuB_lb_90       nuB_ub_90        nuF_lb_90       nuF_ub_90       TB_lb_90        TB_ub_90        TF_lb_90        TF_ub_90        misid_lb_90     misid_ub_90
0.4146571254600148      0.3869760036609383      0.5405870676040649      0.6910597085952759      0.01265975832939148     2489.12434337297        0.013825473303355517    12.436502383830257      0.07848449596883701     1.9080255986972294       -0.03580846309661867    1.1169825983047486      0.14834715366363527     1.2337722635269164      -0.014465123414993286   0.039784640073776245    0.005312160541557728    32.367344764837036      0.05010784243050499      2.9885626709447877      -0.19791970610618592    1.2790938413143158      -0.00429075241088861    1.3864101696014404      -0.0220939964056015     0.04741351306438446

# CIs:    |----------80----------|      |----------90----------|
# nuB:    [  0.013825,  12.436502]      [  0.005312,  32.367345]
# nuF:    [  0.078484,   1.908026]      [  0.050108,   2.988563]
# TB:     [ -0.035808,   1.116983]      [ -0.197920,   1.279094]
# TF:     [  0.148347,   1.233772]      [ -0.004291,   1.386410]
# misid:  [ -0.014465,   0.039785]      [ -0.022094,   0.047414]
```

If users have [trained their own MLPRs](#training-custom-mlprs), they can direct donni to the trained MLPRs file directory with the `--mlpr_dir` argument.

```console
$ donni infer --input_fs examples/data/1d_ns20_sfs.fs --model two_epoch --mlpr_dir examples/data/two_epoch_20_mlprs/
```

```console
***Inferred demographic model parameters***
# nu    T       misid   theta   nu_lb_95        nu_ub_95        T_lb_95 T_ub_95 misid_lb_95     misid_ub_95
0.24181668926375313     0.9589159488677979      0.0008354485034942627   4046.0925977734805      0.02504349414857273     2.334950181455216       0.1795339167118073      1.7382979810237884      -0.018864348903298377   0.020535245910286902

# CIs:    |----------95----------|
# nu:     [  0.025043,   2.334950]
# T:      [  0.179534,   1.738298]
# misid:  [ -0.018864,   0.020535]
```

## Undefined theta when an inferred parameter is negative
Sometimes the trained MLPRs will output a negative parameter estimation, leading to an undefined theta value. An example is included and can be accessed by running the following command:

```console
$ donni infer --input_fs examples/data/1d_ns20_sfs.fs --model two_epoch --mlpr_dir examples/data/two_epoch_20_mlprs_negative/
```

When this is the case, the pipeline will output something like the below with a warning about undefined theta:

```console
***Inferred demographic model parameters***
# nu    T       misid   theta   nu_lb_95        nu_ub_95        T_lb_95 T_ub_95 misid_lb_95     misid_ub_95
0.13990851968127813     1.1827903985977173      -0.010053243488073349   nan     0.010451441687755284    1.8728893548092593      0.3536277294158936      2.011953067779541       -0.02659363556653261    0.006487148590385912

# CIs:    |----------95----------|
# nu:     [  0.010451,   1.872889]
# T:      [  0.353628,   2.011953]
# misid:  [ -0.026594,   0.006487]

WARNING: Theta is not defined. Check inferred demographic model parameters for negative values.
```
As indicated in the output here, the misid parameter is very slightly negative (-0.010), causing theta to be undefined. Handling of cases like this depends on a case-by-case basis, such as which parameter is negative, how accurate the trained MLPRs are on predicting such parameter (should be reviewed in the uploaded QC validation plots), and the absolute value of the negative estimation. In this example, it is likely that misid is simply very close to 0, which is good, but further optimization with dadi/dadi-cli is recommended.

# Training custom MLPRs
The three subcommands `generate_data`, `train`, and `validate` are for training and testing trained MLPRs. This is the procedure we used to produce all the MLPRs in the current library. Users can use the same method outlined below to create their custom demographic model with dadi and produce the corresponding trained MLPRs.

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


## Generating data: full example commands
To generate 5000 training AFS for the out_of_africa model found in the donni/donni/custom_models.py file and the hyperparmeters:

```console
$ donni generate_data --model out_of_africa --model_file donni/custom_models.py --n_samples 5000 \
--sample_sizes 10 10 10 --seed 1 --outfile data/train_5000
```
To generate 1000 test AFS with moderate noise for the same model:

```console
$ donni generate_data --model out_of_africa --model_file donni/custom_models.py --n_samples 1000 \
--sample_sizes 10 10 10 --seed 100 --theta 1000 --outfile data/test_1000_theta_1000
```

## Hyperparameter tuning and training the MLPR
​After we have generated the data for training and testing, we will now use these data to tune and train the MLPRs for each demographic model parameter. This can be done using the donni subcommand `train` with the two required flags: `--data_file` pointing to the training data output from the previous step, and `--mlpr_dir` indicating the path the save the output trained MLPR.

```console
$ donni train --data_file data/train_5000 --mlpr_dir trained_models
```

While it is possible to train MLPR using the default set of hyperparameters, we recommend users to first run the tuning procedure to find the most optimized set of hyperparameters. This can be done by adding the argument `--tune`.

```console
$ donni train --data_file data/train_5000 --mlpr_dir tuned_models --tune
```

## Validating trained MLPRs accuracy and confidence interval coverage
Finally, we can use the simulated test data to measure the accuracy performance of the trained MLPRs with the subcommand `validate`. The required arguments are:

`--mlpr_dir`: path to the trained MLPRs to be validated

`--test_dict`: path to the test data

`--results_dir`: path to save the output plots

`--plot_prefix`: for the prefix of each plot's filename

`--model`: for the demographic model used (required to obtain the demographic model parameters). If a custom model is used then also need to provide a path to it with `--model_file`.

`--folded`: required if the MLPRs were trained on folded AFS data (no misid parameter in demographic model).

Below is an example of a full command:
```console
$ donni validate --model two_epoch --mlpr_dir examples/data/two_epoch_20_mlprs \
--test_dict examples/data/two_epoch_test_1000_theta_1000 \
--results_dir examples/data/plots --plot_prefix two_epoch_theta_1000
```

### Description of arguments:
For descriptions of all arguments, use:
```console
$ donni validate -h
```

# Requirements
1. Python 3.11+
2. [dadi](https://dadi.readthedocs.io/en/latest/)
3. [Tensorflow](https://www.tensorflow.org/) 2.12
4. [keras-tuner](https://keras.io/keras_tuner/) 1.4.6


# References
1. [Tran et al., bioRxiv, 2023.](https://www.biorxiv.org/content/10.1101/2023.05.24.542158v2)
2. [Huang et al., bioRxiv, 2023.](https://www.biorxiv.org/content/10.1101/2023.06.15.545182v1.abstract)
3. [Gutenkunst et al., *PLoS Genet*, 2009.](https://journals.plos.org/plosgenetics/article?id=10.1371/journal.pgen.1000695)
