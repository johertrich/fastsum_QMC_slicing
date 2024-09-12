# script for generating the plots for the kernel summation

import h5py
import numpy as np
import matplotlib.pyplot as plt
import os


def gen_plot(kernel_nr, dataset_nr):
    names = dict(
        RFF="Monte Carlo RFF",
        ORF="ORF",
        QMRFFS="QMC RFF-10 Slicing",
        MCS="Monte Carlo Fourier Slicing",
        QMCFS="QMC Fourier Slicing",
        SOBOLRFF="QMC (Sobol) RFF",
    )
    if kernel_nr == 4:
        names["MCS"] = "Monte Carlo Sorting Slicing"
        names["QMCFS"] = "QMC Sorting Slicing"

    markers = dict(RFF="v", ORF="^", SOBOLRFF="<", QMRFFS="*", MCS="x", QMCFS="s")
    colors = dict(RFF="y", ORF="k", SOBOLRFF="c", QMRFFS="m", MCS="b", QMCFS="r")

    if kernel_nr == 0:
        kernel_name = "Gauss kernel"
    elif kernel_nr == 1:
        kernel_name = r"Matern ($\nu=3.5$)"
    elif kernel_nr == 2:
        kernel_name = r"Matern ($\nu=1.5$)"
    elif kernel_nr == 3:
        kernel_name = "Laplace kernel"
    elif kernel_nr == 4:
        kernel_name = "negative distance"

    if dataset_nr == 0:
        dataset_name = "Letters ($d=16$)"
    elif dataset_nr == 1:
        dataset_name = "MNIST ($d=20$)"
    elif dataset_nr == 2:
        dataset_name = "FashionMNIST ($d=30$)"
    elif dataset_nr == 3:
        dataset_name = "MNIST ($d=784$)"
    elif dataset_nr == 4:
        dataset_name = "FashionMNIST ($d=784$)"

    title = kernel_name + " on " + dataset_name

    if kernel_nr == 0:
        methods = ["RFF", "ORF", "SOBOLRFF", "QMRFFS", "MCS", "QMCFS"]
    elif kernel_nr <= 3:
        methods = ["RFF", "ORF", "QMRFFS", "MCS", "QMCFS"]
    else:
        methods = ["MCS", "QMCFS"]

    errors_dict = dict()
    times_dict = dict()

    fname = (
        "../results/results_ds_"
        + str(dataset_nr)
        + "_kernel_"
        + str(kernel_nr)
        + "_errors.h5"
    )
    with h5py.File(fname, "r") as f:
        for method in methods:
            errors_dict[method] = f[method][()]

    fname = (
        "../results/results_ds_"
        + str(dataset_nr)
        + "_kernel_"
        + str(kernel_nr)
        + "_times.h5"
    )
    with h5py.File(fname, "r") as f:
        for method in methods:
            times_dict[method] = f[method][()]

    fig = plt.figure(figsize=(6, 6 / 8 * 6))
    for method in methods:
        errors_mean = np.mean(errors_dict[method], 0)
        errors_std = np.std(errors_dict[method], 0)
        times_mean = np.mean(times_dict[method], 0)
        print(method, "errors", errors_mean)
        print(method, "times", times_mean)
        plt.loglog(
            times_mean,
            errors_mean,
            colors[method] + markers[method],
            label=names[method],
        )
        plt.fill_between(
            times_mean,
            errors_mean - errors_std,
            errors_mean + errors_std,
            alpha=0.1,
            color=colors[method],
        )
    plt.xlabel("Time (s) on single-threaded CPU", fontsize=12)
    plt.ylabel("Relative $L^1$ Error", fontsize=12)
    plt.title(title, fontsize=15)
    if dataset_nr <= 2:
        plt.xticks(ticks=np.array([0.1, 1, 10]), fontsize=12)
    plt.yticks(fontsize=12)
    plt.rcParams["xtick.direction"] = "out"
    plt.rcParams["ytick.direction"] = "out"
    plt.legend(loc="lower left", prop={"size": 12})
    if not os.path.isdir("../figs"):
        os.mkdir("../figs")
    fig.savefig(
        f"../figs/fig_kernel_{kernel_nr}_ds_{dataset_nr}.pdf", bbox_inches="tight"
    )
    plt.close(fig)


for kernel_nr in [0, 1, 2, 3, 4]:
    for dataset_nr in [4]:
        gen_plot(kernel_nr, dataset_nr)
