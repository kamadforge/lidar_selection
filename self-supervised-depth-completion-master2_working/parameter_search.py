import subprocess
import os



for lr in [0.0001, 1e-5]:
    for wd in [0.0, 0.1, 0.01]:
        for iter in [10000, 20000, 30000, 40000, 45000]:

            print(f"\nlr: {lr} and wd: {wd} iter: {iter}\n")

            path_name = f"/home/kadamczewski/Dropbox_from/Current_research/depth_completion_opt/results/mode=dense.input=gd.resnet34.criterion=l2.lr={lr}.bs=1.wd={wd}.pretrained=False.jitter=0.1.time=2021-05-20@20-05/checkpoint_qnet-0_i_{iter}_typefeature_sq.pth.tar"

            file_name = os.path.split(path_name)[1]

            folder_and_name = path_name.split(os.sep)[-2:]


            subprocess.call(["python", "main_sw.py",
            "--data-folder",  "/home/kamil/Dropbox/Current_research/data/kitti",
            "-e", path_name])


            subprocess.call(["python", "main_orig.py",
            "--data-folder",  "/home/kamil/Dropbox/Current_research/data/kitti",
            "-e", "/home/kamil/Dropbox/Current_research/depth_completion_opt/results/good/mode=dense.input=gd.resnet34.criterion=l2.lr=1e-05.bs=1.wd=0.pretrained=False.jitter=0.1.time=2021-04-01@19-36/checkpoint--1_i_16600_typefeature_None.pth.tar", "--ranks_file"] + folder_and_name)