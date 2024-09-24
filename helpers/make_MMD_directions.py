# brute force optimization for computing the distance MMD directions on the GPU
# using PyKeOps

import torch
from tqdm import tqdm
import pykeops.torch
import os
import h5py

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def interaction_derivative(points):
    q_k = pykeops.torch.LazyTensor(points.unsqueeze(1).contiguous())
    p_k = pykeops.torch.LazyTensor(points.unsqueeze(0).contiguous())
    diff = q_k - p_k
    dist = (diff**2).sum(2).sqrt() + 1e-13
    grad = diff / dist
    out = grad.sum(1)
    return -out


def compute_low_discrepancy_points(
    N, d, n_iter=10000, lr=1e-0, init=None, dtype=torch.float
):
    if init is None:
        points = torch.randn((N, d), device=device, dtype=dtype)
        points = points / torch.sqrt(torch.sum(points**2, -1, keepdim=True))
    else:
        points = init.clone()
    for it in tqdm(range(n_iter)):
        points = points.detach().requires_grad_(True)
        points_big = torch.cat((points, -points), 0)
        grad_big = interaction_derivative(points_big)
        grad = grad_big[:N] - grad_big[N:]
        points = points - lr * grad
        points = points / torch.sqrt(torch.sum(points**2, -1, keepdim=True))
        inds = points[:, 0] < 0
        points[inds] = -points[inds]
        lr = 0.999 * lr
    return points.detach().float()


ds = [3, 5, 10, 16, 20, 30, 50, 100, 200, 784]
Ps = [5 * 2**k for k in range(11)]
if not os.path.isdir("../distance_MMD_projs"):
    os.mkdir("../distance_MMD_projs")
for d in ds:
    for P in Ps:
        only_single = (
            d > 200
        )  # use only single precision whenever the dimension is larger than 200
        xis = compute_low_discrepancy_points(P, d)
        if not only_single:  # go to double precision and reduce learning rate
            xis = compute_low_discrepancy_points(
                P, d, init=xis.to(torch.float64), dtype=torch.float64, lr=1e-1
            )
        else:  # only reduce learning rate
            xis = compute_low_discrepancy_points(P, d, init=xis, lr=1e-1)

        path = "../distance_MMD_projs/d" + str(d)
        if not os.path.isdir(path):
            os.mkdir(path)
        path = path + "/" + "P_sym" + str(P) + ".h5"
        with h5py.File(path, "w") as f:
            f.create_dataset("xis", data=xis.detach().cpu().numpy())
