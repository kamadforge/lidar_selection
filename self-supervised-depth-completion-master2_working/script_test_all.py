import subprocess
import socket
import itertools
import numpy as np




feat_number=32

def test_switch(checkpoint_path, rank_file, input_form):

    if socket.gethostname()=='kamilblade':
        subprocess.run(
            ["python", "main_orig_jul26.py", "--data-folder", "/home/kamil/Dropbox/Current_research/data/kitti", "--i", input_form, "--epochs", "20", "-e", checkpoint_path, "--seed", "121", "--type_feature", "lines","--test_mode", "switch", "--feature_mode", "global", "--feature_num", str(feat_number), "--rank_file_global_sq", rank_file])
        #subprocess.run(["python", "main_orig_jul26.py", "--data-folder",  "/home/kamil/Dropbox/Current_research/data/kitti", "--i", input_form, "--epochs", "20", "-e", checkpoint_path,  "--seed", "121", "--type_feature", "sq", "--test_mode", "switch", "--feature_mode", "global",  "--feature_num", feat_number, "--rank_file_global_sq", rank_file])
    else:
        subprocess.run(["/home/kadamczewski/miniconda3/bin/python", "/home/kadamczewski/Dropbox_from/Current_research/depth_completion/self-supervised-depth-completion-master2_working/main_orig_jul26.py", "--data-folder", "/is/cluster/scratch/kamil/kitti", "--i", input_form, "--type_feature", "sq", "--train-mode", "dense", "--workers", "0", "-e" ,checkpoint_path, "--seed", "121", "--test_mode", "switch", "--feature_mode", "global",  "--feature_num", str(feat_number), "--rank_file_global_sq", rank_file])


def test_features_in_checkpoint(checkpoint_path, input_form):

    test_modes = [ "spaced", "most", "random"]
    #test_modes = ["custom"]
    #test_modes = []
    #for i in range(65):
    #    test_modes.append("one"+str(i))
    
    #twos = list(itertools.combinations(np.arange(65), 2))
    #for t in twos:
    #    test_modes.append("two"+str(t))
    
        
    feat_modes = ["global", "local"]
    #feat_modes = ["global"]
    feat_nums = [feat_number]

    for tm in test_modes:
        for fm in feat_modes:
            for fn in feat_nums:
                print("\n\n{tm}  {tf}  {fn}\n\n")
                subprocess.run(["/home/kadamczewski/miniconda3/bin/python", "/home/kadamczewski/Dropbox_from/Current_research/depth_completion/self-supervised-depth-completion-master2_working/main_orig.py", "--data-folder", "/is/cluster/scratch/kamil/kitti", "--i", input_form, "--type_feature", "lines", "--train-mode", "dense", "--workers", "0", "-e" ,checkpoint_path, "--seed", "121", "--test_mode", tm, "--feature_mode", fm,  "--feature_num", str(fn)])
