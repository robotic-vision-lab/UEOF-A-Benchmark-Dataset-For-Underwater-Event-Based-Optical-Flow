#!/bin/bash

scenes=(scena1 scena2 scena3 scena4 scena5)

for scene in "${scenes[@]}"; do
  python v2e.py -i "/v2e/input/shallow/${scene}/output.mp4" \
    --output_folder="output/shallow/${scene}" \
    --dvs_h5 "${scene}.hdf5" \
    --output_width=960 \
    --output_height=540 \
    --overwrite \
    --no_preview \
    --dvs_exposure source \
    --cutoff_hz=100 \
    --sigma_thres=0.015 \
    --pos_thres=0.16 \
    --neg_thres=0.16 \
    --timestamp_resolution=0.002 \
    --batch_size=8
done


