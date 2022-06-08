import subprocess
import socket

if socket.gethostname() == "kamilblade":

    # for i in range(10):
    #     subprocess.call(["python", "main_orig.py", "--data-folder", "/home/kamil/Dropbox/Current_research/data/kitti"])

    subprocess.call(
        ["python", "main_orig.py", "--data-folder", "/home/kamil/Dropbox/Current_research/data/kitti", "--test_mode",
         "shap", "--feature_mode", "local", "--feature_num", "32"])

    subprocess.call(
        ["python", "main_orig.py", "--data-folder", "/home/kamil/Dropbox/Current_research/data/kitti", "--test_mode",
         "shap", "--feature_mode", "local", "--feature_num", "16"])

    subprocess.call(
        ["python", "main_orig.py", "--data-folder", "/home/kamil/Dropbox/Current_research/data/kitti", "--test_mode",
         "shap", "--feature_mode", "local", "--feature_num", "8"])

    subprocess.call(
        ["python", "main_orig.py", "--data-folder", "/home/kamil/Dropbox/Current_research/data/kitti", "--test_mode",
         "shap", "--feature_mode", "local", "--feature_num", "4"])

    subprocess.call(
        ["python", "main_orig.py", "--data-folder", "/home/kamil/Dropbox/Current_research/data/kitti", "--test_mode",
         "shap", "--feature_mode", "global", "--feature_num", "32"])

    subprocess.call(
        ["python", "main_orig.py", "--data-folder", "/home/kamil/Dropbox/Current_research/data/kitti", "--test_mode",
         "shap", "--feature_mode", "global", "--feature_num", "16"])

    subprocess.call(
        ["python", "main_orig.py", "--data-folder", "/home/kamil/Dropbox/Current_research/data/kitti", "--test_mode",
         "shap", "--feature_mode", "global", "--feature_num", "8"])

    subprocess.call(
        ["python", "main_orig.py", "--data-folder", "/home/kamil/Dropbox/Current_research/data/kitti", "--test_mode",
         "shap", "--feature_mode", "global", "--feature_num", "4"])

    subprocess.call(
        ["python", "main_orig.py", "--data-folder", "/home/kamil/Dropbox/Current_research/data/kitti", "--test_mode",
         "spaced", "--feature_mode", "global", "--feature_num", "32"])

    subprocess.call(
        ["python", "main_orig.py", "--data-folder", "/home/kamil/Dropbox/Current_research/data/kitti", "--test_mode",
         "spaced", "--feature_mode", "global", "--feature_num", "16"])

    subprocess.call(
        ["python", "main_orig.py", "--data-folder", "/home/kamil/Dropbox/Current_research/data/kitti", "--test_mode",
         "spaced", "--feature_mode", "global", "--feature_num", "8"])

    subprocess.call(
        ["python", "main_orig.py", "--data-folder", "/home/kamil/Dropbox/Current_research/data/kitti", "--test_mode",
         "spaced", "--feature_mode", "global", "--feature_num", "4"])

    subprocess.call(
        ["python", "main_orig.py", "--data-folder", "/home/kamil/Dropbox/Current_research/data/kitti", "--test_mode",
         "random", "--feature_mode", "global", "--feature_num", "32"])

    subprocess.call(
        ["python", "main_orig.py", "--data-folder", "/home/kamil/Dropbox/Current_research/data/kitti", "--test_mode",
         "random", "--feature_mode", "global", "--feature_num", "16"])

    subprocess.call(
        ["python", "main_orig.py", "--data-folder", "/home/kamil/Dropbox/Current_research/data/kitti", "--test_mode",
         "random", "--feature_mode", "global", "--feature_num", "8"])

    subprocess.call(
        ["python", "main_orig.py", "--data-folder", "/home/kamil/Dropbox/Current_research/data/kitti", "--test_mode",
         "random", "--feature_mode", "global", "--feature_num", "4"])

else:

    # for i in range(10000):
    #    print(i)
    #    subprocess.run(["/home/kadamczewski/miniconda3/bin/python", "/home/kadamczewski/Dropbox_from/Current_research/depth_completion/self-supervised-depth-completion-master2_working/main_orig.py", "--data-folder", "/is/cluster/scratch/kamil/kitti", "-e", "/home/kadamczewski/Dropbox_from/Current_research/depth_completion/results/train/2022_02_23_17:45_all_depth_scratch/mode=dense.input=gd.resnet18.criterion=l2.lr=1e-05.bs=1.wd=0.pretrained=False.jitter=0.1.time=2022-02-23@17-46/checkpoint_10_i_85000__best.pth.tar", "--layers", "18"])

    # ALL

    subprocess.run(["/home/kadamczewski/miniconda3/bin/python",
                    "/home/kadamczewski/Dropbox_from/Current_research/depth_completion/self-supervised-depth-completion-master2_working/main_orig.py",
                    "--data-folder", "/is/cluster/scratch/kamil/kitti", "-e",
                    "/home/kadamczewski/Dropbox_from/Current_research/depth_completion/results/train/2022_02_23_17:45_all_depth_scratch/mode=dense.input=gd.resnet18.criterion=l2.lr=1e-05.bs=1.wd=0.pretrained=False.jitter=0.1.time=2022-02-23@17-46/checkpoint_10_i_85000__best.pth.tar",
                    "--layers", "18", "--test_mode", "all", "--feature_mode", "local", "--feature_num", "32"])

    # LOCAL

    subprocess.run(["/home/kadamczewski/miniconda3/bin/python",
                    "/home/kadamczewski/Dropbox_from/Current_research/depth_completion/self-supervised-depth-completion-master2_working/main_orig.py",
                    "--data-folder", "/is/cluster/scratch/kamil/kitti", "-e",
                    "/home/kadamczewski/Dropbox_from/Current_research/depth_completion/results/train/2022_02_23_17:45_all_depth_scratch/mode=dense.input=gd.resnet18.criterion=l2.lr=1e-05.bs=1.wd=0.pretrained=False.jitter=0.1.time=2022-02-23@17-46/checkpoint_10_i_85000__best.pth.tar",
                    "--layers", "18", "--test_mode", "shap", "--feature_mode", "local", "--feature_num", "32"])

    subprocess.run(["/home/kadamczewski/miniconda3/bin/python",
                    "/home/kadamczewski/Dropbox_from/Current_research/depth_completion/self-supervised-depth-completion-master2_working/main_orig.py",
                    "--data-folder", "/is/cluster/scratch/kamil/kitti", "-e",
                    "/home/kadamczewski/Dropbox_from/Current_research/depth_completion/results/train/2022_02_23_17:45_all_depth_scratch/mode=dense.input=gd.resnet18.criterion=l2.lr=1e-05.bs=1.wd=0.pretrained=False.jitter=0.1.time=2022-02-23@17-46/checkpoint_10_i_85000__best.pth.tar",
                    "--layers", "18", "--test_mode", "shap", "--feature_mode", "local", "--feature_num", "16"])

    subprocess.run(["/home/kadamczewski/miniconda3/bin/python",
                    "/home/kadamczewski/Dropbox_from/Current_research/depth_completion/self-supervised-depth-completion-master2_working/main_orig.py",
                    "--data-folder", "/is/cluster/scratch/kamil/kitti", "-e",
                    "/home/kadamczewski/Dropbox_from/Current_research/depth_completion/results/train/2022_02_23_17:45_all_depth_scratch/mode=dense.input=gd.resnet18.criterion=l2.lr=1e-05.bs=1.wd=0.pretrained=False.jitter=0.1.time=2022-02-23@17-46/checkpoint_10_i_85000__best.pth.tar",
                    "--layers", "18", "--test_mode", "shap", "--feature_mode", "local", "--feature_num", "8"])

    subprocess.run(["/home/kadamczewski/miniconda3/bin/python",
                    "/home/kadamczewski/Dropbox_from/Current_research/depth_completion/self-supervised-depth-completion-master2_working/main_orig.py",
                    "--data-folder", "/is/cluster/scratch/kamil/kitti", "-e",
                    "/home/kadamczewski/Dropbox_from/Current_research/depth_completion/results/train/2022_02_23_17:45_all_depth_scratch/mode=dense.input=gd.resnet18.criterion=l2.lr=1e-05.bs=1.wd=0.pretrained=False.jitter=0.1.time=2022-02-23@17-46/checkpoint_10_i_85000__best.pth.tar",
                    "--layers", "18", "--test_mode", "shap", "--feature_mode", "local", "--feature_num", "4"])

    # LOCAL REGION

    subprocess.run(["/home/kadamczewski/miniconda3/bin/python",
                    "/home/kadamczewski/Dropbox_from/Current_research/depth_completion/self-supervised-depth-completion-master2_working/main_orig.py",
                    "--data-folder", "/is/cluster/scratch/kamil/kitti", "-e",
                    "/home/kadamczewski/Dropbox_from/Current_research/depth_completion/results/train/2022_02_23_17:45_all_depth_scratch/mode=dense.input=gd.resnet18.criterion=l2.lr=1e-05.bs=1.wd=0.pretrained=False.jitter=0.1.time=2022-02-23@17-46/checkpoint_10_i_85000__best.pth.tar",
                    "--layers", "18", "--test_mode", "shap", "--feature_mode", "local", "--feature_num", "32",
                    '--region_shap', "1"])

    subprocess.run(["/home/kadamczewski/miniconda3/bin/python",
                    "/home/kadamczewski/Dropbox_from/Current_research/depth_completion/self-supervised-depth-completion-master2_working/main_orig.py",
                    "--data-folder", "/is/cluster/scratch/kamil/kitti", "-e",
                    "/home/kadamczewski/Dropbox_from/Current_research/depth_completion/results/train/2022_02_23_17:45_all_depth_scratch/mode=dense.input=gd.resnet18.criterion=l2.lr=1e-05.bs=1.wd=0.pretrained=False.jitter=0.1.time=2022-02-23@17-46/checkpoint_10_i_85000__best.pth.tar",
                    "--layers", "18", "--test_mode", "shap", "--feature_mode", "local", "--feature_num", "16",
                    '--region_shap', "1"])

    subprocess.run(["/home/kadamczewski/miniconda3/bin/python",
                    "/home/kadamczewski/Dropbox_from/Current_research/depth_completion/self-supervised-depth-completion-master2_working/main_orig.py",
                    "--data-folder", "/is/cluster/scratch/kamil/kitti", "-e",
                    "/home/kadamczewski/Dropbox_from/Current_research/depth_completion/results/train/2022_02_23_17:45_all_depth_scratch/mode=dense.input=gd.resnet18.criterion=l2.lr=1e-05.bs=1.wd=0.pretrained=False.jitter=0.1.time=2022-02-23@17-46/checkpoint_10_i_85000__best.pth.tar",
                    "--layers", "18", "--test_mode", "shap", "--feature_mode", "local", "--feature_num", "8",
                    '--region_shap', "1"])

    subprocess.run(["/home/kadamczewski/miniconda3/bin/python",
                    "/home/kadamczewski/Dropbox_from/Current_research/depth_completion/self-supervised-depth-completion-master2_working/main_orig.py",
                    "--data-folder", "/is/cluster/scratch/kamil/kitti", "-e",
                    "/home/kadamczewski/Dropbox_from/Current_research/depth_completion/results/train/2022_02_23_17:45_all_depth_scratch/mode=dense.input=gd.resnet18.criterion=l2.lr=1e-05.bs=1.wd=0.pretrained=False.jitter=0.1.time=2022-02-23@17-46/checkpoint_10_i_85000__best.pth.tar",
                    "--layers", "18", "--test_mode", "shap", "--feature_mode", "local", "--feature_num", "4",
                    '--region_shap', "1"])

    # LOCAL SEPARATION

    subprocess.run(["/home/kadamczewski/miniconda3/bin/python",
                    "/home/kadamczewski/Dropbox_from/Current_research/depth_completion/self-supervised-depth-completion-master2_working/main_orig.py",
                    "--data-folder", "/is/cluster/scratch/kamil/kitti", "-e",
                    "/home/kadamczewski/Dropbox_from/Current_research/depth_completion/results/train/2022_02_23_17:45_all_depth_scratch/mode=dense.input=gd.resnet18.criterion=l2.lr=1e-05.bs=1.wd=0.pretrained=False.jitter=0.1.time=2022-02-23@17-46/checkpoint_10_i_85000__best.pth.tar",
                    "--layers", "18", "--test_mode", "shap", "--feature_mode", "local", "--feature_num", "32",
                    '--separation_shap', "1"])

    subprocess.run(["/home/kadamczewski/miniconda3/bin/python",
                    "/home/kadamczewski/Dropbox_from/Current_research/depth_completion/self-supervised-depth-completion-master2_working/main_orig.py",
                    "--data-folder", "/is/cluster/scratch/kamil/kitti", "-e",
                    "/home/kadamczewski/Dropbox_from/Current_research/depth_completion/results/train/2022_02_23_17:45_all_depth_scratch/mode=dense.input=gd.resnet18.criterion=l2.lr=1e-05.bs=1.wd=0.pretrained=False.jitter=0.1.time=2022-02-23@17-46/checkpoint_10_i_85000__best.pth.tar",
                    "--layers", "18", "--test_mode", "shap", "--feature_mode", "local", "--feature_num", "16",
                    '--separation_shap', "1"])

    subprocess.run(["/home/kadamczewski/miniconda3/bin/python",
                    "/home/kadamczewski/Dropbox_from/Current_research/depth_completion/self-supervised-depth-completion-master2_working/main_orig.py",
                    "--data-folder", "/is/cluster/scratch/kamil/kitti", "-e",
                    "/home/kadamczewski/Dropbox_from/Current_research/depth_completion/results/train/2022_02_23_17:45_all_depth_scratch/mode=dense.input=gd.resnet18.criterion=l2.lr=1e-05.bs=1.wd=0.pretrained=False.jitter=0.1.time=2022-02-23@17-46/checkpoint_10_i_85000__best.pth.tar",
                    "--layers", "18", "--test_mode", "shap", "--feature_mode", "local", "--feature_num", "8",
                    '--separation_shap', "1"])

    subprocess.run(["/home/kadamczewski/miniconda3/bin/python",
                    "/home/kadamczewski/Dropbox_from/Current_research/depth_completion/self-supervised-depth-completion-master2_working/main_orig.py",
                    "--data-folder", "/is/cluster/scratch/kamil/kitti", "-e",
                    "/home/kadamczewski/Dropbox_from/Current_research/depth_completion/results/train/2022_02_23_17:45_all_depth_scratch/mode=dense.input=gd.resnet18.criterion=l2.lr=1e-05.bs=1.wd=0.pretrained=False.jitter=0.1.time=2022-02-23@17-46/checkpoint_10_i_85000__best.pth.tar",
                    "--layers", "18", "--test_mode", "shap", "--feature_mode", "local", "--feature_num", "4",
                    '--separation_shap', "1"])

    # GLOBAL

    subprocess.run(["/home/kadamczewski/miniconda3/bin/python",
                    "/home/kadamczewski/Dropbox_from/Current_research/depth_completion/self-supervised-depth-completion-master2_working/main_orig.py",
                    "--data-folder", "/is/cluster/scratch/kamil/kitti", "-e",
                    "/home/kadamczewski/Dropbox_from/Current_research/depth_completion/results/train/2022_02_23_17:45_all_depth_scratch/mode=dense.input=gd.resnet18.criterion=l2.lr=1e-05.bs=1.wd=0.pretrained=False.jitter=0.1.time=2022-02-23@17-46/checkpoint_10_i_85000__best.pth.tar",
                    "--layers", "18", "--test_mode", "shap", "--feature_mode", "global", "--feature_num", "32"])

    subprocess.run(["/home/kadamczewski/miniconda3/bin/python",
                    "/home/kadamczewski/Dropbox_from/Current_research/depth_completion/self-supervised-depth-completion-master2_working/main_orig.py",
                    "--data-folder", "/is/cluster/scratch/kamil/kitti", "-e",
                    "/home/kadamczewski/Dropbox_from/Current_research/depth_completion/results/train/2022_02_23_17:45_all_depth_scratch/mode=dense.input=gd.resnet18.criterion=l2.lr=1e-05.bs=1.wd=0.pretrained=False.jitter=0.1.time=2022-02-23@17-46/checkpoint_10_i_85000__best.pth.tar",
                    "--layers", "18", "--test_mode", "shap", "--feature_mode", "global", "--feature_num", "16"])

    subprocess.run(["/home/kadamczewski/miniconda3/bin/python",
                    "/home/kadamczewski/Dropbox_from/Current_research/depth_completion/self-supervised-depth-completion-master2_working/main_orig.py",
                    "--data-folder", "/is/cluster/scratch/kamil/kitti", "-e",
                    "/home/kadamczewski/Dropbox_from/Current_research/depth_completion/results/train/2022_02_23_17:45_all_depth_scratch/mode=dense.input=gd.resnet18.criterion=l2.lr=1e-05.bs=1.wd=0.pretrained=False.jitter=0.1.time=2022-02-23@17-46/checkpoint_10_i_85000__best.pth.tar",
                    "--layers", "18", "--test_mode", "shap", "--feature_mode", "global", "--feature_num", "8"])

    subprocess.run(["/home/kadamczewski/miniconda3/bin/python",
                    "/home/kadamczewski/Dropbox_from/Current_research/depth_completion/self-supervised-depth-completion-master2_working/main_orig.py",
                    "--data-folder", "/is/cluster/scratch/kamil/kitti", "-e",
                    "/home/kadamczewski/Dropbox_from/Current_research/depth_completion/results/train/2022_02_23_17:45_all_depth_scratch/mode=dense.input=gd.resnet18.criterion=l2.lr=1e-05.bs=1.wd=0.pretrained=False.jitter=0.1.time=2022-02-23@17-46/checkpoint_10_i_85000__best.pth.tar",
                    "--layers", "18", "--test_mode", "shap", "--feature_mode", "global", "--feature_num", "4"])

    # SPACED

    subprocess.run(["/home/kadamczewski/miniconda3/bin/python",
                    "/home/kadamczewski/Dropbox_from/Current_research/depth_completion/self-supervised-depth-completion-master2_working/main_orig.py",
                    "--data-folder", "/is/cluster/scratch/kamil/kitti", "-e",
                    "/home/kadamczewski/Dropbox_from/Current_research/depth_completion/results/train/2022_02_23_17:45_all_depth_scratch/mode=dense.input=gd.resnet18.criterion=l2.lr=1e-05.bs=1.wd=0.pretrained=False.jitter=0.1.time=2022-02-23@17-46/checkpoint_10_i_85000__best.pth.tar",
                    "--layers", "18", "--test_mode", "spaced", "--feature_mode", "global", "--feature_num", "32"])

    subprocess.run(["/home/kadamczewski/miniconda3/bin/python",
                    "/home/kadamczewski/Dropbox_from/Current_research/depth_completion/self-supervised-depth-completion-master2_working/main_orig.py",
                    "--data-folder", "/is/cluster/scratch/kamil/kitti", "-e",
                    "/home/kadamczewski/Dropbox_from/Current_research/depth_completion/results/train/2022_02_23_17:45_all_depth_scratch/mode=dense.input=gd.resnet18.criterion=l2.lr=1e-05.bs=1.wd=0.pretrained=False.jitter=0.1.time=2022-02-23@17-46/checkpoint_10_i_85000__best.pth.tar",
                    "--layers", "18", "--test_mode", "random", "--feature_mode", "global", "--feature_num", "32"])

    subprocess.run(["/home/kadamczewski/miniconda3/bin/python",
                    "/home/kadamczewski/Dropbox_from/Current_research/depth_completion/self-supervised-depth-completion-master2_working/main_orig.py",
                    "--data-folder", "/is/cluster/scratch/kamil/kitti", "-e",
                    "/home/kadamczewski/Dropbox_from/Current_research/depth_completion/results/train/2022_02_23_17:45_all_depth_scratch/mode=dense.input=gd.resnet18.criterion=l2.lr=1e-05.bs=1.wd=0.pretrained=False.jitter=0.1.time=2022-02-23@17-46/checkpoint_10_i_85000__best.pth.tar",
                    "--layers", "18", "--test_mode", "spaced", "--feature_mode", "global", "--feature_num", "16"])

    subprocess.run(["/home/kadamczewski/miniconda3/bin/python",
                    "/home/kadamczewski/Dropbox_from/Current_research/depth_completion/self-supervised-depth-completion-master2_working/main_orig.py",
                    "--data-folder", "/is/cluster/scratch/kamil/kitti", "-e",
                    "/home/kadamczewski/Dropbox_from/Current_research/depth_completion/results/train/2022_02_23_17:45_all_depth_scratch/mode=dense.input=gd.resnet18.criterion=l2.lr=1e-05.bs=1.wd=0.pretrained=False.jitter=0.1.time=2022-02-23@17-46/checkpoint_10_i_85000__best.pth.tar",
                    "--layers", "18", "--test_mode", "spaced", "--feature_mode", "global", "--feature_num", "4"])

    # RANDOM

    subprocess.run(["/home/kadamczewski/miniconda3/bin/python",
                    "/home/kadamczewski/Dropbox_from/Current_research/depth_completion/self-supervised-depth-completion-master2_working/main_orig.py",
                    "--data-folder", "/is/cluster/scratch/kamil/kitti", "-e",
                    "/home/kadamczewski/Dropbox_from/Current_research/depth_completion/results/train/2022_02_23_17:45_all_depth_scratch/mode=dense.input=gd.resnet18.criterion=l2.lr=1e-05.bs=1.wd=0.pretrained=False.jitter=0.1.time=2022-02-23@17-46/checkpoint_10_i_85000__best.pth.tar",
                    "--layers", "18", "--test_mode", "random", "--feature_mode", "global", "--feature_num", "16"])

    subprocess.run(["/home/kadamczewski/miniconda3/bin/python",
                    "/home/kadamczewski/Dropbox_from/Current_research/depth_completion/self-supervised-depth-completion-master2_working/main_orig.py",
                    "--data-folder", "/is/cluster/scratch/kamil/kitti", "-e",
                    "/home/kadamczewski/Dropbox_from/Current_research/depth_completion/results/train/2022_02_23_17:45_all_depth_scratch/mode=dense.input=gd.resnet18.criterion=l2.lr=1e-05.bs=1.wd=0.pretrained=False.jitter=0.1.time=2022-02-23@17-46/checkpoint_10_i_85000__best.pth.tar",
                    "--layers", "18", "--test_mode", "spaced", "--feature_mode", "global", "--feature_num", "8"])

    subprocess.run(["/home/kadamczewski/miniconda3/bin/python",
                    "/home/kadamczewski/Dropbox_from/Current_research/depth_completion/self-supervised-depth-completion-master2_working/main_orig.py",
                    "--data-folder", "/is/cluster/scratch/kamil/kitti", "-e",
                    "/home/kadamczewski/Dropbox_from/Current_research/depth_completion/results/train/2022_02_23_17:45_all_depth_scratch/mode=dense.input=gd.resnet18.criterion=l2.lr=1e-05.bs=1.wd=0.pretrained=False.jitter=0.1.time=2022-02-23@17-46/checkpoint_10_i_85000__best.pth.tar",
                    "--layers", "18", "--test_mode", "random", "--feature_mode", "global", "--feature_num", "8"])

    subprocess.run(["/home/kadamczewski/miniconda3/bin/python",
                    "/home/kadamczewski/Dropbox_from/Current_research/depth_completion/self-supervised-depth-completion-master2_working/main_orig.py",
                    "--data-folder", "/is/cluster/scratch/kamil/kitti", "-e",
                    "/home/kadamczewski/Dropbox_from/Current_research/depth_completion/results/train/2022_02_23_17:45_all_depth_scratch/mode=dense.input=gd.resnet18.criterion=l2.lr=1e-05.bs=1.wd=0.pretrained=False.jitter=0.1.time=2022-02-23@17-46/checkpoint_10_i_85000__best.pth.tar",
                    "--layers", "18", "--test_mode", "random", "--feature_mode", "global", "--feature_num", "4"])


