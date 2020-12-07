import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision.models import resnet
import matplotlib.pyplot as plt
from torch.nn.parameter import Parameter

torch.set_printoptions(profile="full")

lines=np.load("../kitti_pixels_to_lines.npy")
parameter_mask = torch.tensor(lines)


parameter = Parameter(-1e-10 * torch.ones(65), requires_grad=True)
phi = F.softplus(parameter)


if any(torch.isnan(phi)):
    print("some Phis are NaN")
# it looks like too large values are making softplus-transformed values very large and returns NaN.
# this occurs when optimizing with a large step size (or/and with a high momentum value)


S = phi / torch.sum(phi)


S_mask_ext = torch.einsum("i, ijk->ijk", [S, parameter_mask])
print(S_mask_ext[24][308][733])
S_mask=torch.max(S_mask_ext, 0)[0]
S_mask_ind=torch.max(S_mask_ext, 0)[1]
#print(S_mask_ind[300])

for i in range(len(parameter)):
    print(f"{i}: {len(np.where(S_mask_ind==i)[0])}")

a=0