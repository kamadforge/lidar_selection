#!/bin/bash

echo 'Save path is: '$1
echo 'Data path is: '${4-/esat/rat/wvangans/Datasets/KITTI/Depth_Completion2/Data/}

python Test/test.py --save_path $1 --num_samples ${2-0} --mod ${3-"mod"} --data_path ${4-/esat/rat/wvangans/Datasets/KITTI/Depth_Completion2/Data/} --layers 18 --normal true

# Arguments for evaluate_depth file: 
# - ground truth directory
# - results directory

Test/devkit/cpp/evaluate_depth ${5-/esat/rat/wvangans/Datasets/KITTI/Depth_Completion/data/depth_selection/val_selection_cropped/groundtruth_depth} $1/results 
