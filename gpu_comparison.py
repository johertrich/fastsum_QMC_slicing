import torch
import torchvision.datasets as td
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from python_src.fastsum import *
from python_src.basis_funs import Gaussian_kernel_fun_ft, Matern_kernel_fun_ft
import time
from python_src.utils import *
import pickle
from simple_torch_NFFT import NFFT, GaussWindow
from scipy.stats import qmc, norm
import scipy
import numpy as np
import os
import h5py

torch._dynamo.config.cache_size_limit = 160

import argparse

parser = argparse.ArgumentParser(description="Choose parameters")
parser.add_argument(
    "--kernel_nr", type=int, default=0
)  # numbering for several runs, can be None
inp = parser.parse_args()

kernel_nr = inp.kernel_nr

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##################
# create dataset #
##################
mnist = td.MNIST("mnist", transform=transforms.ToTensor(), download=True)
data = DataLoader(dataset=mnist, batch_size=60000)
X = next(iter(data))[0].view(60000, -1)
X = X - X.mean(0, keepdims=True)
X = X.view(60000, 28, 28)
X = torch.cat((X,torch.rot90(X,dims=[-2,-1]),torch.rot90(X,k=2,dims=[-2,-1]) ,torch.rot90(X,k=3,dims=[-2,-1])),0)
X = torch.cat((X, X.transpose(-2, -1)), 0).view(-1, 28**2)
X = X.view(-1, 28**2)
X = X.detach().cpu().numpy()

fmnist = td.FashionMNIST("fmnist", transform=transforms.ToTensor(), download=True)
data = DataLoader(dataset=fmnist, batch_size=60000)
X2 = next(iter(data))[0].view(60000, -1)
X2 = X2 - X2.mean(0, keepdims=True)
X2 = X2.view(60000, 28, 28)
X2 = torch.cat((X2,torch.rot90(X2,dims=[-2,-1]),torch.rot90(X2,k=2,dims=[-2,-1]) ,torch.rot90(X2,k=3,dims=[-2,-1])),0)
X2 = torch.cat((X2, X2.transpose(-2, -1)), 0).view(-1, 28**2)
X2 = X2.view(-1, 28**2)
X2 = X2.detach().cpu().numpy()

X = np.concatenate((X, X2), 0)


dims = 30

U, s, Vh = scipy.linalg.svd(X.transpose(), full_matrices=False)

mat = U[:, :dims]
X_down = X @ mat

x = torch.tensor(X_down, device=device, dtype=torch.float)
y = torch.tensor(X_down, device=device, dtype=torch.float)
x_weights = torch.ones(x.shape[0]).to(x)

##########################
# RFF sampling functions #
##########################


def sample_multivariate_t(n, d, nu):
    loc = np.zeros((d,))
    shape = np.eye(d)
    out = scipy.stats.multivariate_t.rvs(loc=loc, shape=shape, df=nu, size=(n,))
    return out


def sample_1d_Gaussian(n, d):
    return np.sqrt(np.random.chisquare(d, size=n))


def sample_1d_Matern(n, d, nu):
    out = sample_multivariate_t(n, d, 2 * nu)
    return np.sqrt((out**2).sum(-1))


##############
# parameters #
##############

runs = 10
med = get_median_distance(x, y)
sliced_factor = compute_sliced_factor(dims)
d = x.shape[1]
n_ft = 1024
batch_size_nfft = 2  # worked best
batch_size_P = 640  # chosen as large as possible such that no memory error appears.
soboleng = qmc.Sobol(d=dims, scramble=True)


# create NFFT
nfft = NFFT((n_ft,), m=2, grad_via_adjoint=False, no_compile=False)


keops_batch_size = x.shape[0]
if kernel_nr == 0:
    kernel_fun = lambda x, y, scale: (-0.5 * ((x - y) ** 2).sum(-1) / scale**2).exp()
    fourier_fun = lambda x, scale: Gaussian_kernel_fun_ft(x, d, scale**2)
if kernel_nr == 1:
    nu = torch.tensor(3.5).to(x)

    def kernel_fun(x, y, scale):
        arg = torch.sqrt(2 * nu) * ((x - y) ** 2).sum(-1).sqrt() / scale
        return (1 + arg + 0.4 * arg**2 + (arg**3) / 15) * (-arg).exp()

    fourier_fun = lambda x, scale: Matern_kernel_fun_ft(x, d, scale, 3.5)
    # keops_batch_size=60000
if kernel_nr == 2:
    nu = torch.tensor(1.5).to(x)

    def kernel_fun(x, y, scale):
        arg = torch.sqrt(2 * nu) * ((x - y) ** 2).sum(-1).sqrt() / scale
        return (1 + arg) * (-arg).exp()

    fourier_fun = lambda x, scale: Matern_kernel_fun_ft(x, d, scale, 1.5)
    # keops_batch_size=60000
if kernel_nr == 3:
    kernel_fun = lambda x, y, scale: (-((x - y) ** 2).sum(-1).sqrt() / scale).exp()
    fourier_fun = lambda x, scale: Matern_kernel_fun_ft(x, d, scale, 0.5)
if kernel_nr == 4:
    kernel_fun = lambda x, y, scale: -((x - y) ** 2).sum(-1).sqrt()

###########
# PyKeOps #
###########

torch.cuda.empty_cache()
# compile keops
naive_kernel_sum_keops(x, x_weights, y, med, kernel_fun, keops_batch_size)
torch.cuda.empty_cache()

print("Compile done")


print("Keops")
torch.cuda.synchronize()
time.sleep(0.5)
tic = time.time()
s_true = naive_kernel_sum_keops(x, x_weights, y, med, kernel_fun, keops_batch_size)
torch.cuda.synchronize()
toc = time.time() - tic
time_keops = toc

s_true = torch.tensor(s_true.detach().cpu().numpy()).to(x)
torch.cuda.empty_cache()


#########################################################
# Function for runnin all methods using some specific P #
#########################################################

def test(P):
    print(f"P={P}")
    errors_rff = None
    mean_time_rff = None
    errors_orf = None
    mean_time_orf = None
    errors_qmc_rff = None
    mean_time_qmc_rff = None
    if kernel_nr <= 3:
        P_rff = 4 * P
        torch._dynamo.reset()
        errors_rff = []
        times_rff = []
        print("RFF")
        for _ in range(runs + 1):
            torch.cuda.empty_cache()
            if kernel_nr == 0:
                rff_xis = torch.randn(P_rff, d).to(x)
            if kernel_nr == 1:
                rff_xis = torch.tensor(sample_multivariate_t(P_rff, d, 7)).to(x)
            if kernel_nr == 2:
                rff_xis = torch.tensor(sample_multivariate_t(P_rff, d, 3)).to(x)
            if kernel_nr == 3:
                rff_xis = torch.tensor(sample_multivariate_t(P_rff, d, 1)).to(x)
            torch.cuda.synchronize()
            time.sleep(0.5)
            tic = time.time()
            s_rff = RFF(x, y, x_weights, med, min(batch_size_P, P_rff), rff_xis)
            torch.cuda.synchronize()
            toc = time.time() - tic
            times_rff.append(toc)
            errors_rff.append(
                (
                    torch.sum(torch.abs(s_rff - s_true)) / torch.sum(torch.abs(s_true))
                ).item()
            )
        # don't count the first run since GPUs are often require time to initialize/compile subroutines...
        times_rff = times_rff[1:]
        errors_rff = errors_rff[1:]
        mean_time_rff = np.mean(times_rff)
        print(f"Mean time {mean_time_rff}")

        errors_orf = []
        times_orf = []
        print("ORF")
        for _ in range(runs + 1):
            torch.cuda.empty_cache()
            orf_xis = np.random.normal(size=(P_rff // d + 1, d, d))
            orf_xis, _ = np.linalg.qr(orf_xis)
            orf_xis_orth = orf_xis.reshape(-1, d)[:P_rff]
            if kernel_nr == 0:
                orf_xis = torch.tensor(
                    orf_xis_orth.copy() * sample_1d_Gaussian(P_rff, d)[:, None]
                ).to(x)
            if kernel_nr == 1:
                orf_xis = torch.tensor(
                    orf_xis_orth.copy() * sample_1d_Matern(P_rff, d, 3.5)[:, None]
                ).to(x)
            if kernel_nr == 2:
                orf_xis = torch.tensor(
                    orf_xis_orth.copy() * sample_1d_Matern(P_rff, d, 1.5)[:, None]
                ).to(x)
            if kernel_nr == 3:
                orf_xis = torch.tensor(
                    orf_xis_orth.copy() * sample_1d_Matern(P_rff, d, 0.5)[:, None]
                ).to(x)
            torch.cuda.synchronize()
            time.sleep(0.5)
            tic = time.time()
            s_orf = RFF(x, y, x_weights, med, min(batch_size_P, P_rff), orf_xis)
            torch.cuda.synchronize()
            toc = time.time() - tic
            times_orf.append(toc)
            errors_orf.append(
                (
                    torch.sum(torch.abs(s_orf - s_true)) / torch.sum(torch.abs(s_true))
                ).item()
            )
        # don't count the first run since GPUs are often require time to initialize/compile subroutines...
        times_orf = times_orf[1:]
        errors_orf = errors_orf[1:]
        mean_time_orf = np.mean(times_orf)
        print(f"Mean time {mean_time_orf}")

        torch._dynamo.reset()
        if kernel_nr == 0:
            torch.cuda.empty_cache()
            errors_qmc_rff = []
            times_qmc_rff = []
            P_sobol = int(2 ** np.floor(np.log2(P_rff)))
            batch_size_P_sobol = int(2 ** np.floor(np.log2(batch_size_P)))
            torch.cuda.empty_cache()
            theta = soboleng.random(n=P_sobol)
            theta = np.clip(theta, a_min=1e-6, a_max=1 - 1e-6)
            theta = norm.ppf(theta)
            xis_sobol = torch.tensor(theta).to(x)
            print("QMC-RFF")
            for _ in range(runs + 1):
                orth_mat = np.random.normal(size=(d, d))
                orth_mat = torch.tensor(np.linalg.qr(orth_mat)[0]).to(x)
                xis_qmc_rff = xis_sobol @ orth_mat
                torch.cuda.synchronize()
                time.sleep(0.5)
                tic = time.time()
                s_rff = RFF(
                    x, y, x_weights, med, min(batch_size_P_sobol, P_sobol), xis_qmc_rff
                )
                torch.cuda.synchronize()
                toc = time.time() - tic
                times_qmc_rff.append(toc)
                errors_qmc_rff.append(
                    (
                        torch.sum(torch.abs(s_rff - s_true))
                        / torch.sum(torch.abs(s_true))
                    ).item()
                )
            times_qmc_rff = times_qmc_rff[1:]
            errors_qmc_rff = errors_qmc_rff[1:]
            mean_time_qmc_rff = np.mean(times_qmc_rff)
            print(f"Mean time {mean_time_qmc_rff}")

    torch._dynamo.reset()
    path = "distance_MMD_projs/d" + str(d) + "/" + "P_sym" + str(P) + ".h5"
    f = h5py.File(path, 'r')
    xis_base = torch.tensor(f["xis"][()]).to(x)

    errors_mmd = []
    times_mmd = []
    print("QMC-Slicing")
    for _ in range(runs + 1):
        torch.cuda.empty_cache()
        orth_mat = np.random.normal(size=(d, d))
        orth_mat = torch.tensor(np.linalg.qr(orth_mat)[0]).to(x)
        xis = xis_base @ orth_mat
        xis = xis.to(device)
        torch.cuda.synchronize()
        time.sleep(0.5)
        tic = time.time()
        if kernel_nr <= 3:
            s = fastsum_fft(x,y,x_weights,med,n_ft,.3,fourier_fun,xis,nfft=nfft,batch_size_P=min(batch_size_P*2,P),batch_size_nfft=batch_size_nfft)
        else:
            s = fast_energy_summation(
                x, y, x_weights, sliced_factor, min(batch_size_P // 4, P), xis
            )
        torch.cuda.synchronize()
        toc = time.time() - tic
        times_mmd.append(toc)
        errors_mmd.append(
            (torch.sum(torch.abs(s - s_true)) / torch.sum(torch.abs(s_true))).item()
        )
    # don't count the first run since GPUs are often require time to initialize/compile subroutines...
    times_mmd = times_mmd[1:]
    errors_mmd = errors_mmd[1:]
    mean_time_mmd = np.mean(times_mmd)
    print(f"Mean time {mean_time_mmd}")

    errors_sl = []
    times_sl = []
    print("Slicing")
    for _ in range(runs + 1):
        torch.cuda.empty_cache()
        xis = torch.randn(P, d).to(x)
        xis = xis / torch.sqrt(torch.sum(xis**2, -1, keepdims=True))
        torch.cuda.synchronize()
        time.sleep(0.5)
        tic = time.time()
        if kernel_nr <= 3:
            s = fastsum_fft(x,y,x_weights,med,n_ft,.3,fourier_fun,xis,nfft=nfft,batch_size_P=min(batch_size_P,P),batch_size_nfft=batch_size_nfft)
        else:
            s = fast_energy_summation(
                x, y, x_weights, sliced_factor, min(batch_size_P // 4, P), xis
            )
        torch.cuda.synchronize()
        toc = time.time() - tic
        times_sl.append(toc)
        errors_sl.append(
            (torch.sum(torch.abs(s - s_true)) / torch.sum(torch.abs(s_true))).item()
        )
    # don't count the first run since GPUs are often require time to initialize/compile subroutines...
    times_sl = times_sl[1:]
    errors_sl = errors_sl[1:]
    mean_time_sl = np.mean(times_sl)
    print(f"Mean time {mean_time_sl}")
    return [
        mean_time_rff,
        mean_time_orf,
        mean_time_qmc_rff,
        mean_time_mmd,
        mean_time_sl,
    ], [errors_rff, errors_orf, errors_qmc_rff, errors_mmd, errors_sl]


###################################################
# Calling comparison, printing and saving results #
###################################################

print("Time PyKeOps:", time_keops)
times_all = []
errors_all = []
methods = ["RFF", "ORF", "QMC-RFF", "QMC Slicing", "Slicing"]
for P in [10 * 2**k for k in range(10)]:
    times, errors = test(P)
    print("\nMethods:",[method for method,time in zip(methods,times) if time is not None])
    print("Times:",[time for time in times if time is not None])
    print("Mean Errors:", [np.mean(errs) for errs in errors if errs is not None])
    print("Std Errors:", [np.std(errs) for errs in errors if errs is not None],"\n")
    times_all.append([t for t in times if t is not None])
    errors_all.append([e for e in errors if e is not None])

if not os.path.isdir("gpu_results"):
    os.mkdir("gpu_results")
path = f"gpu_results/results_kernel_{kernel_nr}.pickle"
with open(path, "wb") as f:
    pickle.dump(
        dict(
            times=np.array(times_all),
            errors=np.array(errors_all),
            time_keops=time_keops,
        ),
        f,
    )
