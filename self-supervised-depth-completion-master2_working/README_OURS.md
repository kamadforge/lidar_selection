Pipeline:

1. We train the main model from scratch (sparse or not sparse)
2. We train the switches
3. We choose a ranking (switches or another method, e..g random) and test

-----

Training:
train the model from scratch (main_orig.py)
--data-folder /home/kamil/Dropbox/Current_research/data/kitti --epochs 20

train the model from scratch on a subset of features, currently only squares (main_orig_trainsparse.py)
--data-folder /home/kamil/Dropbox/Current_research/data/kitti --epochs 20

train the switches main_sw.py
--data-folder /home/kamil/Dropbox/Current_research/data/kitti --resume *checkpoint_trained_from_scratch* --epochs 20

Test:
All tests are done on main_orig.py.
--data-folder /home/kamil/Dropbox/Current_research/data/kitti --epochs 20 -e *checkpoint_trained_from_scratch*
can test on all the features or using the subset of features, the switch rank would be read from file.

-----

Saved results:
checkpoints are saved in depth../results/
the ranks are saved in depth../self../ranks/
the photos are saved in depth../self../switch_photos


Other crucial files used:

dataloaders/kitti_loader (different versions) - loads the depth file
features/depth_manipulation - selects subsets of depth features
features/depth_draw - saves the rgb images with features 

-----

You can train the original model from scratch but here are pretrained model examples:

trained sparse:
https://www.dropbox.com/s/ao7vxyg9yqraz6q/checkpoint--1_i_16600_typefeature_None.pth.tar?dl=0

trained dense:
https://www.dropbox.com/s/fzatriljr6eq6nj/checkpoint_qnet-9_i_0_typefeature_None.pth.tar?dl=0



