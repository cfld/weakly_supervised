#!/usr/bin/env bash

for i in 0 1 2 3 4 5
do
    for n in 10 20 40 80 100
    do
        for p in 64 128 256
        do
        CUDA_VISIBLE_DEVICES=4 python run_masked.py \
                        --param_file /home/ebarnett/weakly_supervised_future/single_pixel_labels/experiments/params_naip.json \
                        --model_dir /home/ebarnett/weakly_supervised_future/single_pixel_labels/experiments/ \
                        --results_dir /home/ebarnett/weakly_supervised_future/single_pixel_labels/experiments/naip_results/ \
                        --seed $i --n_train $n --label_size $p


        CUDA_VISIBLE_DEVICES=4 python run_masked.py \
                        --param_file /home/ebarnett/weakly_supervised_future/single_pixel_labels/experiments/params_naip.json \
                        --model_dir /home/ebarnett/weakly_supervised_future/single_pixel_labels/experiments/ \
                        --results_dir /home/ebarnett/weakly_supervised_future/single_pixel_labels/experiments/naip_results/ \
                        --seed $i --n_train $n --label_size $p --pretrained

        done
    done
done