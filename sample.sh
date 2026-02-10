#!/bin/bash

THRESHOLD_LIST=(
    1
)

NUM_REFINE_LIST=(
    1
)

TASK_LIST=(
    # deblur_gauss
    # deblur_motion
    # super_resolution
    # random_inpainting
    # box_inpainting
    temporal_avg
)

# Main Loop
for task in "${TASK_LIST[@]}"; do
    for threshold in "${THRESHOLD_LIST[@]}"; do
        for refine_count in "${NUM_REFINE_LIST[@]}"; do
            echo "================================================================"
            echo "Running: task=${task}, ths=${threshold}, refine=${refine_count}"
            echo "================================================================"

            OUTPUT_DIR="videos/self_forcing_dmd"

            python inference_restoration.py \
                --config_path configs/self_forcing_dmd.yaml \
                --output_folder "${OUTPUT_DIR}" \
                --checkpoint_path checkpoints/self_forcing_dmd.pt \
                --data_path /mnt/storage/projects/Self-Forcing/data/pexels \
                --task "${task}" \
                --use_ema \
                --restoration \
                --ths_uncertainty "${threshold}" \
                --num_refine "${refine_count}"
        done
    done
done

echo "All inference tasks are completed!"