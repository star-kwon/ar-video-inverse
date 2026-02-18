#!/bin/bash

THRESHOLD_LIST=(
    0.5
)

NUM_REFINE_LIST=(
    3
)

TASK_LIST=(
    deblur_gauss
    deblur_motion
    super_resolution
    random_inpainting
    temporal_avg
)

TASKS_STR="${TASK_LIST[*]}"

for threshold in "${THRESHOLD_LIST[@]}"; do
    for refine_count in "${NUM_REFINE_LIST[@]}"; do
        echo "================================================================"
        echo "Running: tasks=[${TASKS_STR}], ths=${threshold}, refine=${refine_count}"
        echo "================================================================"

        OUTPUT_DIR="videos/self_forcing_dmd"

        python inference_restoration.py \
            --config_path configs/self_forcing_dmd.yaml \
            --output_folder "${OUTPUT_DIR}" \
            --checkpoint_path checkpoints/self_forcing_dmd.pt \
            --data_path /mnt/storage/projects/Self-Forcing/data/samples \
            --task_list ${TASK_LIST[@]} \
            --use_ema \
            --restoration \
            --ths_uncertainty "${threshold}" \
            --num_refine "${refine_count}"
    done
done

echo "All inference tasks are completed!"