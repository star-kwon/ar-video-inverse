#!/bin/bash

INITIALIZATION_LIST=(
    100
)

SAMPLING_LIST=(
    2
)

TASK_LIST=(
    deblur_gauss
    random_inpainting
    super_resolution
    temporal_avg
    spatio_temporal_avg
)

TASKS_STR="${TASK_LIST[*]}"

for initialization_step in "${INITIALIZATION_LIST[@]}"; do
    for sampling_step in "${SAMPLING_LIST[@]}"; do
        echo "================================================================"
        echo "Running: tasks=[${TASKS_STR}], initialization_step=${initialization_step}, sampling_step=${sampling_step}"
        echo "================================================================"

        OUTPUT_DIR="results"

        python inference_restoration.py \
            --config_path configs/self_forcing_dmd.yaml \
            --output_folder "${OUTPUT_DIR}" \
            --checkpoint_path checkpoints/self_forcing_dmd.pt \
            --data_path /mnt/storage/projects/Self-Forcing/data/pexels \
            --task_list ${TASK_LIST[@]} \
            --use_ema \
            --restoration \
            --initialization_step "${initialization_step}" \
            --sampling_step "${sampling_step}"
    done
done

echo "All inference tasks are completed!"