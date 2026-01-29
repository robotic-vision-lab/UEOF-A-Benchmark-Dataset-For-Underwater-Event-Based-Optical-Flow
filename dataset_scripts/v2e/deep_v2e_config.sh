#!/bin/bash

scenes=(s1 s2 s3 s4 s5)

for scene in "${scenes[@]}"; do
  python v2e.py -i "/v2e/input/deep/${scene}.mp4" \
    --output_folder="output/deep/${scene}" \
    --dvs_h5 "${scene}.hdf5" \
    --output_width=1280 \
    --output_height=720 \
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
