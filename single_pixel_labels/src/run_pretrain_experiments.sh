#!/usr/bin/env bash

for i in 0 1 2 3 4 5
do
    for n in 10 20 40 80 100
    do
        for p in 64 128 256
        do
        CUDA_VISIBLE_DEVICES=3 python run_masked.py \
                        --model_dir /home/ebarnett/weakly_supervised/single_pixel_labels/experiments/ \
                        --results_dir /home/ebarnett/weakly_supervised/single_pixel_labels/experiments/results/ \
                        --seed $i --n_train $n --label_size $p


        CUDA_VISIBLE_DEVICES=3 python run_masked.py \
                        --model_dir /home/ebarnett/weakly_supervised/single_pixel_labels/experiments/ \
                        --results_dir /home/ebarnett/weakly_supervised/single_pixel_labels/experiments/results/ \
                        --seed $i --n_train $n --label_size $p --pretrained

        done
    done
done