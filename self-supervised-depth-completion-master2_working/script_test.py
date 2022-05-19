import subprocess

for i in range(10):
    subprocess.call(["python", "main_orig.py", "--data-folder", "/home/kamil/Dropbox/Current_research/data/kitti"])