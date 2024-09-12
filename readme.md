# Fast Summation of Radial Kernels via QMC Slicing

This repository contains the implementations for the paper "Fast Summation of Radial Kernels via QMC Slicing".
If there are any questions, please do not hesitate to contact us.

## Installation

To install all dependencies simply run

```
julia install.jl
```

## Usage

Generally, the fast summation via slicing is implemented in the function `fastsum_fft` from `src/fastsum.jl`. For instance, in the case
of the Gaussian kernel, the fast summation can be performed as follows.

```julia
using HDF5

include("src/fastsum.jl")
include("src/basis_functions.jl")
include("src/utils.jl")

N=100000 # number of data points
M=N # number of data points
d=10 # dimension
P=640 # number of directions
fourier_fun(h,scale) = Gaussian_kernel_fun_ft(h,d,scale^2) # implementation of the Fourier transform of the 1D basis function f
# fourier_fun(h,scale) = Matern_kernel_fun_ft(h,d,scale,nu) # for Matern (and Laplace with nu=0.5)

# generate data
x=randn(N,d)
y=randn(N,d)
x_weights=ones(N)

# median rule
sigma=median_distance(x,y,100)

# load QMC directions
fid = h5open("distance_MMD_projs/d"*string(d)*"/P_sym"*string(P)*".h5","r")
xis=reshape(collect(fid["xis"]),d,P)'

# Parameters
x_range = 0.3 # threshhold T from the rescaling of the 1D kernel summation (for Matern 0.2, Laplace 0.1 worked fine)
n_ft = 128 # number of Fourier coefficients for the NFFT (for Matern 512, Laplace 1024 worked fine)

# summation
s = fastsum_fft(x,y,x_weights,sigma,n_ft,x_range,fourier_fun,xis)
# sliced_factor = compute_sliced_factor(d) # for the negative distance kernel
# s = fastsum_energy(x,y,x_weights,sliced_factor,xis) # fast summation with negative distance kernel
```

The scripts `basis_function_approximation.jl` and `summation_comparison.jl` reproduce the examples from Section 4.2 and 4.3 from the paper.
In order to make the run times reproducible, we ran the time comparisons in `summation_comparison.jl` on a single CPU thread using `hwloc-bind`.

For generating the QMC directions and plotting the results, we provide helper functions written in Python in the directory `helpers`.

## Citation

Anonymized for double blind review.
