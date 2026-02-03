#!/bin/bash

NUM_REFINE_LIST=(1 3 5)
THS_UNCERTAINTY_LIST=(0.25 0.5 1.0)

for num in "${NUM_REFINE_LIST[@]}"; do
    for ths in "${THS_UNCERTAINTY_LIST[@]}"; do
        echo "----------------------------------------------------------------"
        echo "Running Inference: num_refine=${num}, ths_uncertainty=${ths}"
        echo "----------------------------------------------------------------"

        python inference_restoration.py \
            --config_path configs/self_forcing_dmd.yaml \
            --output_folder videos/self_forcing_dmd \
            --checkpoint_path checkpoints/self_forcing_dmd.pt \
            --data_path /mnt/storage/projects/Self-Forcing/data/pexels \
            --use_ema \
            --restoration \
            --num_refine "$num" \
            --ths_uncertainty "$ths"
    done
done