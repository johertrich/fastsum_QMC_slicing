# script for generating the plots for the basis function approximation

import numpy as np
import h5py
import os
import matplotlib.pyplot as plt


def estimate_rate(Ps, errors):
    logP = np.log10(Ps)
    log_errors = np.log10(errors)

    y_bar = np.mean(log_errors)
    x_bar = np.mean(logP)
    ls_a = np.sum((log_errors - y_bar) * (logP - x_bar)) / np.sum((logP - x_bar) ** 2)
    ls_b = y_bar - ls_a * x_bar
    return ls_a, ls_b


def plot(kernel_nr, kernel_factor, dim):

    if kernel_nr == 4 and kernel_factor != 1.0:
        return

    methods = ["RFF", "ORF", "Sobol-RFF", "MC", "Sobol", "orth", "MMD", "SD"]
    names = [
        "RFF",
        "ORF",
        "Sobol RFF",
        "slicing",
        "Sobol slicing",
        "Orth slicing",
        "Distance",
        "spherical design",
    ]
    colors = ["y", "k", "c", "b", "g", "m", "r", "k"]
    markers = ["v", "^", "<", "x", "o", ">", "s", "+"]
    names_dict = {}
    colors_dict = {}
    markers_dict = {}
    for method, name, color, marker in zip(methods, names, colors, markers):
        names_dict[method] = name
        colors_dict[method] = color
        markers_dict[method] = marker
    rff_add = ""
    if kernel_nr == 0:
        rff_add = "Gauss"
    elif kernel_nr == 1:
        rff_add = "Matern3"
    elif kernel_nr == 2:
        rff_add = "Matern1"
    elif kernel_nr == 3:
        rff_add = "Laplace"

    used_methods = []

    errors_dict = {}
    fname = (
        "qmc_comp/qmc_comp_dim_"
        + str(dim)
        + "_kernel_"
        + str(kernel_nr)
        + "_kernel_factor_"
        + str(kernel_factor)
        + ".h5"
    )
    with h5py.File(fname, "r") as f:
        Ps_base = f["Ps"][()]
        Ps_sobol = f["Ps-sobol"][()]
        Ps_sd = f["Ps-sd"][()]
        scale = f["scale"][()]

        for method in methods:
            name = method
            if name[:3] == "RFF" or name[:3] == "ORF":
                name = name + "-" + rff_add
            if name in f.keys():
                used_methods.append(method)
                errors_dict[method] = f[name][()]
    if kernel_nr == 0:
        title = r"Gauss kernel with $\sigma^2={0:.3f}$, $d={1}$".format(scale**2, dim)
        save_name = "Gauss"
    elif kernel_nr == 1:
        title = r"Matern kernel with $\beta={0:.2f}$, $\nu={1:.1f}$, $d={2}$".format(
            scale, 3.5, dim
        )
        save_name = "Matern3"
    elif kernel_nr == 2:
        title = r"Matern kernel with $\beta={0:.2f}$, $\nu={1:.1f}$, $d={2}$".format(
            scale, 1.5, dim
        )
        save_name = "Matern1"
    elif kernel_nr == 3:
        title = r"Laplace kernel with $\alpha={0:.2f}$, $d={1}$".format(1 / scale, dim)
        save_name = "Laplace"
    elif kernel_nr == 4:
        title = r"negative distance kernel $d={0}$".format(dim)
        save_name = "energy"
    rates = []
    fig = plt.figure()
    for method in used_methods:
        if method in ["Sobol-RFF", "Sobol"]:
            Ps = Ps_sobol
        elif method == "SD":
            Ps = Ps_sd
        else:
            Ps = Ps_base
        logP = np.log10(Ps)
        errors = errors_dict[method]
        ls_a, ls_b = estimate_rate(Ps, errors)
        plt.loglog(
            Ps,
            errors,
            colors_dict[method] + markers_dict[method],
            label=names_dict[method],
        )
        if not (method == "SD" and kernel_nr == 0):
            plt.loglog(Ps, 10 ** (ls_a * logP + ls_b), colors_dict[method] + "-")
        rates = rates + [method + " O(N^{0:.2f})".format(ls_a)]
    plt.xlabel("Number P of Projections", fontsize=12)
    plt.ylabel("Absolute Error", fontsize=12)
    plt.title(title, fontsize=15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(loc="lower left", prop={"size": 12})
    if not os.path.isdir("qmc_figs"):
        os.mkdir("qmc_figs")
    if not os.path.isdir("rates"):
        os.mkdir("rates")
    fig.savefig(
        "qmc_figs/fig_" + save_name + f"_d{dim}_s{kernel_factor}.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)
    with open("rates/rates_" + save_name + f"_d{dim}_s{kernel_factor}.txt", "w") as f:
        f.writelines([rate + "\n" for rate in rates])


for dim in [3, 5, 10, 20, 50, 100, 200]:
    for kernel_nr in [0, 1, 2, 3, 4]:
        for kernel_factor in [0.5, 1.0, 2.0]:
            plot(kernel_nr, kernel_factor, dim)
