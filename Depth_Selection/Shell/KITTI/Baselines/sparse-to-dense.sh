#!/usr/bin/sh
model='sparse'
optimizer='adam'
batch_size=${1-7}
lr=${2-0.001}
lr_policy='plateau'
nepochs=60
patience=5
wrgb=${3-0.1}
drop=${4-0.3}
nsamples=${5-0}
multi=${6-0}

data_path=${9-'/srv/beegfs02/scratch/sensor_fusion/data/Datasets/KITTI/Data/'}
out_dir='/scratch_net/schnauz/patilv/code/kamil/Depth_Selection/Saved/'
export OMP_NUM_THREADS=1
cd /scratch_net/schnauz/patilv/code/kamil/Depth_Selection/

python /scratch_net/schnauz/patilv/code/kamil/Depth_Selection/main.py --mod $model --data_path $data_path --optimizer $optimizer --learning_rate $lr --lr_policy $lr_policy --batch_size $batch_size --nepochs $nepochs --no_tb true --lr_decay_iters $patience --num_samples $nsamples --multi $multi --nworkers 4 --save_path $out_dir --wrgb $wrgb --drop $drop --pretrained 0

echo "python has finisched its "$nepochs" epochs!"
echo "Job finished"
