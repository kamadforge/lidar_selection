import subprocess

def test_switch(checkpoint_path, rank_file, input_form):
    #subprocess.run(["python", "main_orig_jul26.py", "--data-folder",  "/home/kamil/Dropbox/Current_research/data/kitti", "--epochs", "20", "-e", checkpoint_path, "--rank_file_global_sq", rank_file])
    
    subprocess.run(["/home/kadamczewski/miniconda3/bin/python", "/home/kadamczewski/Dropbox_from/Current_research/depth_completion/self-supervised-depth-completion-master2_working/main_orig_jul26.py", "--data-folder", "/is/cluster/scratch/kamil/kitti", "--i", input_form, "--type_feature", "sq", "--train-mode", "dense", "--workers", "0", "-e" ,checkpoint_path, "--seed", "121", "--test_mode", "switch", "--feature_mode", "global",  "--feature_num", "10", "--rank_file_global_sq", rank_file])
    

def test_features_in_checkpoint(checkpoint_path, input_form):

    test_modes = ["most", "random"]
    feat_modes = ["global", "local"]
    feat_nums = [10]

    for tm in test_modes:
        for fm in feat_modes:
            for fn in feat_nums:
                print("\n\n{tm}  {tf}  {fn}\n\n")
                subprocess.run(["/home/kadamczewski/miniconda3/bin/python", "/home/kadamczewski/Dropbox_from/Current_research/depth_completion/self-supervised-depth-completion-master2_working/main_orig_jul26.py", "--data-folder", "/is/cluster/scratch/kamil/kitti", "--i", input_form, "--type_feature", "sq", "--train-mode", "dense", "--workers", "0", "-e" ,checkpoint_path, "--seed", "121", "--test_mode", tm, "--feature_mode", fm,  "--feature_num", str(fn)])
