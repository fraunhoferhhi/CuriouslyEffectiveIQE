#!/bin/bash

# iterate over datasets
for db in "live" "tid2013" "csiq"; do

  if [[ "$db" == "live" ]]; then
    path_csv="/home/becker/data/liveiqa/liveiqa.csv"
  fi

  if [[ "$db" == "csiq" ]]; then
    path_csv="/home/becker/data/csiq/_csiq.csv"
  fi

  if [[ "$db" == "tid2013" ]]; then
    path_csv="/home/becker/data/tid2013/tid2013/_tid2013.csv"
  fi

  if [[ "$model" == "cornia" ]]; then
    path_zca="./codebooks/CSIQ_whitening_param.mat"
  else
    path_zca="False"
  fi

  # iterate over codebook models
  for model in "cornia" "patches" "laplace" "normal" "uniform"; do

    echo "Extracting features from $db with $model ."

    python feature_extraction.py  --path_csv "$path_csv" \
                                  --path_codebook ./codebooks/codebook_"$model".npy \
                                  --random_seed 123 \
                                  --block_size 7 \
                                  --max_blocks 10000 \
                                  --name "$model" \
                                  --path_zca "$path_zca" \
                                  --use_pil_convert "False" \
                                  --use_pytorch "True" \
                                  --path_out "./features.pkl"

  done
done