import subprocess
import os
import sys
print(sys.version)


for i in range(10000):

    seed = i
    print(seed)

    subprocess.call(["/home/kadamczewski/miniconda3/bin/python", "main_orig.py", "--workers", "4", "--data-folder",  "/is/cluster/scratch/kamil/kitti", "-e", "/home/kadamczewski/Dropbox_from/Current_research/depth_completion/results/good/mode=dense.input=gd.resnet34.criterion=l2.lr=1e-05.bs=1.wd=0.pretrained=False.jitter=0.1.time=2021-04-01@19-36/checkpoint--1_i_16600_typefeature_None.pth.tar", "--type_feature", "sq",  "--seed", str(seed)])
