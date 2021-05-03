#!/bin/bash
model='mod'
optimizer='adam'

chars=${#CUDA_VISIBLE_DEVICES}
ngpus=$(((chars+1)/2))
echo "Using "$ngpus" GPUs"

batch_size=${1-7}
lr=${2-0.001}
lr_policy=${3-'plateau'}
nepochs=60
patience=5
wrgb=${4-0.1}
drop=${5-0.3}
nsamples=${6-0}
multi=${7-1}
loss=${8-'mse'}

data_path=${9-'/srv/beegfs02/scratch/sensor_fusion/data/Datasets/KITTI/Data/'}
out_dir='/scratch_net/schnauz/patilv/code/kamil/Depth_Selection/Saved/'
export OMP_NUM_THREADS=1
cd /scratch_net/schnauz/patilv/code/kamil/Depth_Selection/

python -m torch.distributed.launch --nproc_per_node=$ngpus \
    /scratch_net/schnauz/patilv/code/kamil/Depth_Selection/main_distributed.py --mod $model --data_path $data_path --optimizer $optimizer --learning_rate $lr --lr_policy $lr_policy --batch_size $batch_size --nepochs $nepochs --no_tb true --lr_decay_iters $patience --num_samples $nsamples --multi $multi --nworkers 4 --save_path $out_dir --wrgb $wrgb --drop $drop --loss_criterion $loss --lr_policy $lr_policy --world_size $ngpus

echo "python has finisched its "$nepochs" epochs!"
echo "Job finished"
