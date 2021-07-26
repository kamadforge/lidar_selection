import subprocess

test_modes = ["most", "random", "switch"]
feat_modes = ["global", "local"]
feat_nums = [10]

for tm in test_modes:
    for fm in feat_modes:
        for fn in feat_nums:
            print(f"\n\n{tm}  {fm}  {fn}\n\n")
            subprocess.run(["/home/kadamczewski/miniconda3/bin/python", "/home/kadamczewski/Dropbox_from/Current_research/depth_completion/self-supervised-depth-completion-master2_working/main_orig.py", "--data-folder", "/is/cluster/scratch/kamil/kitti", "--i", "rgbd", "--type_feature", "sq", "--train-mode", "dense", "--workers", "0", "-e" ,"/home/kadamczewski/Dropbox_from/Current_research/depth_completion/results/mode=dense.input=rgbd.resnet34.criterion=l2.lr=1e-05.bs=1.wd=0.pretrained=False.jitter=0.1.time=2021-07-25@22-01/checkpoint_qnet--1_i_20000_typefeature_None.pth.tar", "--seed", "121", "--test_mode", tm, "--feature_mode", fm,  "--feature_num", str(fn)])
            
#"/home/kadamczewski/Dropbox_from/Current_research/depth_completion/results/good/mode=dense.input=gd.resnet34.criterion=l2.lr=1e-05.bs=1.wd=0.pretrained=False.jitter=0.1.time=2021-05-24@22-50/checkpoint_qnet-9_i_0_typefeature_None.pth.tar"
