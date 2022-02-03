# it's a file that is similar to the one from publish_dirichlet and shapley_rank.py

from sklearn.linear_model import LinearRegression
import numpy as np
import math

filename = "ranks/lines/global/shap/lines_shap.txt"
filename = "/home/kamil/Dropbox/Current_research/depth_completion_opt/self-supervised-depth-completion-master2_working/ranks/lines/global/shap/lines_shap.txt"
filename_clean = "/home/kamil/Dropbox/Current_research/depth_completion_opt/self-supervised-depth-completion-master2_working/ranks/lines/global/shap/lines_shap2.txt"


#remove the multiple entries
file = open(filename)
s = {}
for line in file:
    split_keyval = line.strip().split(":")
    if len(split_keyval[0])>0: #0 subset
        s[split_keyval[0]]=split_keyval[1]

# make a new file with unrepeated entries
file = open(filename_clean, "w+")
file.truncate()
for el in s:
    file.write(el+":"+s[el]+"\n")
file = open(filename_clean)

# get the dictionary with key - the nodes in a subset, value: set value (the difference)
dic = {}
feat_num=65
NOPRUNING_VAL = 11315
for line in file:
    print(line)
    split_keyval = line.strip().split(":")
    val = float(split_keyval[1])
    key = [int(v) for v in split_keyval[0].split(",")]
    key_bin = np.zeros(feat_num)
    key_bin[key]=1
    dic[tuple(key_bin)]=NOPRUNING_VAL - val
    m=s

#dic, nodes_num = readdata_notsampled(file_old, acc)

def nCr(n,k):
    f = math.factorial
    return f(n) / (f(k) * f(n-k))

# compute the weights from the shapley regression, there is a weight for each data point
weights=[]
for i in dic.keys():
    k = np.sum(i)
    val = (feat_num-1)/(nCr(feat_num, k)*k*(feat_num-k))
    weights.append(val)

reg = LinearRegression().fit(list(dic.keys()), list(dic.values()), weights)
shap_arr = reg.coef_
print("shaps\n", shap_arr)
shap_arr_sorted = np.argsort(shap_arr)[-32:]
print(",".join([str(a) for a in shap_arr_sorted]))