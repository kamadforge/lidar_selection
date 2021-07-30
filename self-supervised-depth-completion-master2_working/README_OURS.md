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

the ranks are saved in ranks/

Test:
All tests are done on main_orig.py.
--data-folder /home/kamil/Dropbox/Current_research/data/kitti --epochs 20 -e *checkpoint_trained_from_scratch*

may read ranks/


