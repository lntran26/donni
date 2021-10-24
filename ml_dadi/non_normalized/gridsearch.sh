#!/bin/bash

DATA_DIR="data/1d_2epoch"
OUT_DIR="results/gridsearch/1d_2epoch"
now="$(date + '%Y%m%d-%H%M%S')"

./gridsearchCV_nn_cli.py $DATA_DIR/train_data_10000 -hls 200 500 -a tanh relu -s lbfgs adam -mi 5000 -v 4 &>$OUT_DIR/1d_2epoch_gridsearch_$now.txt