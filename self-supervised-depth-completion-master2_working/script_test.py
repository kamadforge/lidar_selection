import subprocess
import socket
import numpy as np
from collections import OrderedDict
from indexed import IndexedOrderedDict
import random
import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'

if socket.gethostname() == "kamilblade":
    shap_global_path = "/home/kamil/Dropbox/Current_research/depth_completion_opt/self-supervised-depth-completion-master2_working/ranks/lines/global/shap/shapimp_arr.npy"
else:
    shap_global_path = "/home/kadamczewski/Dropbox_from/Current_research/depth_completion/self-supervised-depth-completion-master2_working/ranks/lines/global/shap/shapimp_arr.npy"

shap_global = np.load(shap_global_path)
shap_tup = []
for i, val in enumerate(shap_global):
    shap_tup.append((i, val))

feature_num = 16
for i in range(24):  # 64-16-24
    print("num: ", i)
    lines = shap_tup[-16 - i:]  # top 16 plus possible spread
    print("len:", len(lines))
    for r in range(i):
        element = random.choice(lines)
        lines.remove(element)

    print(np.sum([it[1] for it in lines]))

    lines_str = ",".join([str(it[0]) for it in lines])
    print(lines_str)

    feature_num = str(feature_num)

    if socket.gethostname() == "kamilblade":

        # for i in range(10):
        #     subprocess.call(["python", "main_orig.py", "--data-folder", "/home/kamil/Dropbox/Current_research/data/kitti"])

        for method in ["custom"]:  # "shap", "spaced", "random", "custom"
            for mode in ["global"]:
                if 1:
                #for feature_num in range(1,65):
                    subprocess.call(["python", "main_orig.py", "--data-folder", "/home/kamil/Dropbox/Current_research/data/kitti", "--test_mode", method, "--feature_mode", mode, "--feature_num", feature_num, "--custom_lines", lines_str])

    else:

        # for i in range(10000):
        #    print(i)
        #    subprocess.run(["/home/kadamczewski/miniconda3/bin/python", "/home/kadamczewski/Dropbox_from/Current_research/depth_completion/self-supervised-depth-completion-master2_working/main_orig.py", "--data-folder", "/is/cluster/scratch/kamil/kitti", "-e", "/home/kadamczewski/Dropbox_from/Current_research/depth_completion/results/train/2022_02_23_17:45_all_depth_scratch/mode=dense.input=gd.resnet18.criterion=l2.lr=1e-05.bs=1.wd=0.pretrained=False.jitter=0.1.time=2022-02-23@17-46/checkpoint_10_i_85000__best.pth.tar", "--layers", "18"])


       for method in ["custom"]:  # "shap", "spaced", "random", "custom"
           for mode in ["global"]:
        # for method in ["shap"]:  # "shap", "spaced", "random"
        #     for mode in ["local"]:
                if 1:
                #for num in range(1,65):
                    # feature_num=str(num)
                    subprocess.run(["/home/kadamczewski/miniconda3/bin/python","/home/kadamczewski/Dropbox_from/Current_research/depth_completion/self-supervised-depth-completion-master2_working/main_orig.py", "--data-folder", "/is/cluster/scratch/kamil/kitti", "-e", "/home/kadamczewski/Dropbox_from/Current_research/depth_completion/results/train/2022_02_23_17:45_all_depth_scratch/mode=dense.input=gd.resnet18.criterion=l2.lr=1e-05.bs=1.wd=0.pretrained=False.jitter=0.1.time=2022-02-23@17-46/checkpoint_10_i_85000__best.pth.tar", "--layers", "18", "--test_mode", method, "--feature_mode", mode, "--feature_num", feature_num, '--region_shap', "0", '--separation_shap', "0", "--custom_lines", lines])


