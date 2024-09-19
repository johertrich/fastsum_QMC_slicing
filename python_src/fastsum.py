import numpy as np
import torch
import pykeops.torch
import pykeops


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fast_fourier_summation(x_proj, y_proj, x_weights, kernel_ft, nfft):
    a = nfft.adjoint(-x_proj, x_weights.reshape(1, 1, -1))
    a_time_kernel = a * kernel_ft
    return torch.mean(torch.real(nfft(-y_proj, a_time_kernel)).squeeze(1), 0)


def fastsum_fft(
    x,
    y,
    x_weights,
    scale,
    n_ft,
    x_range,
    fourier_fun,
    xis,
    nfft,
    batch_size_P=None,
    batch_size_nfft=None,
):

    P = xis.shape[0]
    M = y.shape[0]
    if batch_size_P is None:
        batch_size_P = P
    if batch_size_nfft is None:
        batch_size_nfft = batch_size_P
    batch_size_nfft = min(batch_size_nfft, batch_size_P)

    d = x.shape[1]
    xy_norm = torch.sqrt(torch.sum(torch.cat((x, y), 0) ** 2, -1))
    max_norm = torch.max(xy_norm)

    scale_factor = 0.25 * x_range / max_norm
    scale_max = 0.1
    if scale * scale_factor > scale_max:
        scale_factor = scale_max / scale
    scale_real = scale * scale_factor
    x = x * scale_factor
    y = y * scale_factor
    h = torch.arange((-n_ft + 1) // 2, (n_ft + 1) // 2, device=device)
    kernel_ft = fourier_fun(h, scale_real)  # Gaussian_kernel_fun_ft(h,d,scale_real**2)

    xi = xis.unsqueeze(1)

    def with_projections(xi):
        P_local = xi.shape[0]
        x_proj = (xi @ x.T).reshape(P_local, 1, -1, 1)
        y_proj = (xi @ y.T).reshape(P_local, 1, -1, 1)
        outs = torch.stack(
            [
                fast_fourier_summation(
                    x_proj[i * batch_size_nfft : (i + 1) * batch_size_nfft],
                    y_proj[i * batch_size_nfft : (i + 1) * batch_size_nfft],
                    x_weights,
                    kernel_ft,
                    nfft,
                )
                for i in range(P_local // batch_size_nfft)
            ],
            0,
        )
        return torch.mean(outs, 0)

    outs = torch.stack(
        [
            with_projections(xi[i * batch_size_P : (i + 1) * batch_size_P])
            for i in range(P // batch_size_P)
        ],
        0,
    )
    return torch.mean(outs, 0)


def fastsum_energy_kernel_1D(x, x_weights, y):
    # Sorting algorithm for fast sumation with negative distance (energy) kernel
    N = x.shape[1]
    M = y.shape[1]
    P = x.shape[0]
    # Potential Energy
    sorted_yx, inds_yx = torch.sort(torch.cat((y, x), 1))
    inds_yx = inds_yx + torch.arange(P, device=device).unsqueeze(1) * (N + M)
    inds_yx = torch.flatten(inds_yx)
    weights_sorted = (
        torch.cat((torch.zeros_like(y), x_weights), 1).flatten()[inds_yx].reshape(P, -1)
    )
    pot0 = torch.sum(weights_sorted * (sorted_yx - sorted_yx[:, 0:1]), 1, keepdim=True)
    yx_diffs = sorted_yx[:, 1:] - sorted_yx[:, :-1]
    # Mults from cumsums shifted by 1
    mults_short = (
        torch.sum(x_weights, -1, keepdim=True)
        - 2 * torch.cumsum(weights_sorted, 1)[:, :-1]
    )
    mults = torch.zeros_like(weights_sorted)
    mults[:, 1:] = mults_short
    potential = torch.zeros_like(sorted_yx)
    potential[:, 1:] = yx_diffs.clone()
    potential = pot0 - torch.cumsum(potential * mults, 1)
    out1 = torch.zeros_like(sorted_yx).flatten()
    out1[inds_yx] = potential.flatten()
    out1 = out1.reshape(P, -1)
    out1 = out1[:, :M]
    return out1


def fast_energy_summation(x, y, x_weights, sliced_factor, batch_size, xis):
    # fast sum via slicing and sorting
    d = x.shape[1]
    P = xis.shape[0]

    def with_projections(xi):
        P_local = xi.shape[0]
        x_proj = (xi @ x.T).reshape(P_local, -1)
        y_proj = (xi @ y.T).reshape(P_local, -1)
        fastsum_energy = fastsum_energy_kernel_1D(
            x_proj, x_weights[None, :].tile(P_local, 1), y_proj
        ).transpose(0, 1)
        return sliced_factor * torch.mean(-fastsum_energy, 1)

    return torch.mean(
        torch.stack(
            [
                with_projections(xis[i * batch_size : (i + 1) * batch_size])
                for i in range(P // batch_size)
            ],
            0,
        ),
        0,
    )


def naive_kernel_sum_keops(x, x_weights, y, scale, kernel_fun, batch_size=100):
    # Naive kernel summation via the kernel matrix
    # with pykeops
    d = x.shape[1]
    N = x.shape[0]
    naive_sum = 0
    i = 0
    y = pykeops.torch.LazyTensor(y[None, :, :])
    # batching for fitting memory constraints
    while i < N:
        x_batch = pykeops.torch.LazyTensor(x[i : i + batch_size, None, :])
        x_weights_batch = pykeops.torch.LazyTensor(
            x_weights[i : i + batch_size, None], axis=0
        )
        kernel_mat = kernel_fun(x_batch, y, scale)
        naive_sum += (kernel_mat * x_weights_batch).sum(0)
        i += batch_size
    return naive_sum.squeeze()


def RFF(x, y, x_weights, scale, batch_size, xis):
    # batching over the projections
    out = 0
    d = x.shape[1]
    P = xis.shape[0]

    def with_projections(xi):
        x_proj = (xi @ x.T).transpose(0, 1) / scale
        y_proj = (xi @ y.T).transpose(0, 1) / scale
        cos_x = torch.cos(x_proj)
        cos_y = torch.cos(y_proj)
        cos_x_sum = torch.sum(cos_x * x_weights[:, None], 0)
        sin_x = torch.sin(x_proj)
        sin_y = torch.sin(y_proj)
        sin_x_sum = torch.sum(sin_x * x_weights[:, None], 0)
        res = torch.mean(cos_y * cos_x_sum[None, :], -1) + torch.mean(
            sin_y * sin_x_sum[None, :], -1
        )
        return res / P * batch_size

    return torch.sum(
        torch.stack(
            [
                with_projections(xis[i * batch_size : (i + 1) * batch_size])
                for i in range(P // batch_size)
            ],
            0,
        ),
        0,
    )
