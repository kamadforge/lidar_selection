import subprocess
import os
import sys
print(sys.version)


for lr in [0.0001, 1e-5]:
    for wd in [0.0, 0.1, 0.01]:
        for ep in [0,1,2,3,4]:
            for iter in [50000, 70000, 85500]:
            

                print(f"\nlr: {lr} and wd: {wd} ep: {ep} iter: {iter}\n")

                path_name1 = f"/home/kadamczewski/Dropbox_from/Current_research/depth_completion/results/mode=dense.input=gd.resnet34.criterion=l2.lr={lr}.bs=1.wd={wd}.pretrained=False.jitter=0.1.time=2021-05-21@13-01/checkpoint_qnet-{ep}_i_{iter}_typefeature_sq.pth.tar"
                
                path_name2 = f"/home/kadamczewski/Dropbox_from/Current_research/depth_completion/results/mode=dense.input=gd.resnet34.criterion=l2.lr={lr}.bs=1.wd={wd}.pretrained=False.jitter=0.1.time=2021-05-21@13-02/checkpoint_qnet-{ep}_i_{iter}_typefeature_sq.pth.tar"
                
                if os.path.isfile(path_name1): 
                   path_name = path_name1
                else:
                   path_name = path_name2
                   
                print(path_name)
                
             

                file_name = os.path.split(path_name)[1]

                folder_and_name = path_name.split(os.sep)[-2:]


                subprocess.call(["/home/kadamczewski/miniconda3/bin/python", "main_sw.py", "--workers", "4",
                "--data-folder",  "/is/cluster/scratch/kamil/kitti",
                "-e", path_name])


                subprocess.call(["/home/kadamczewski/miniconda3/bin/python", "main_orig.py", "--workers", "4",
                "--data-folder",  "/is/cluster/scratch/kamil/kitti",
                "-e", "/home/kadamczewski/Dropbox_from/Current_research/depth_completion/results/good/mode=dense.input=gd.resnet34.criterion=l2.lr=1e-05.bs=1.wd=0.pretrained=False.jitter=0.1.time=2021-04-01@19-36/checkpoint--1_i_16600_typefeature_None.pth.tar", "--ranks_file"] + folder_and_name)
