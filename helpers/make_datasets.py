# make datasets for the examples and save them as h5-files

from ucimlrepo import fetch_ucirepo
import h5py
import torchvision.datasets as td
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import scipy
import os


if os.path.isdir("../datasets"):
    os.mkdir("../datasets")

# letter dataset
letter_recognition = fetch_ucirepo(id=59)
X = letter_recognition.data.features.to_numpy()
with h5py.File("../datasets/letter.h5", "w") as f:
    f.create_dataset("data", data=X)

# MNIST
mnist = td.MNIST("mnist", transform=transforms.ToTensor(), download=True)
data = DataLoader(dataset=mnist, batch_size=60000)
X = next(iter(data))[0].view(60000, -1).detach().cpu().numpy()

# centering
X = X - X.mean(0, keepdims=True)

# save full dataset
with h5py.File("../datasets/mnist784.h5", "w") as f:
    f.create_dataset("data", data=X)

# reduce dimension by PCA
dims = 20
U, s, Vh = scipy.linalg.svd(X.transpose(), full_matrices=False)
U = U[:, :dims]
X_down = X @ U
with h5py.File("../datasets/mnist20.h5", "w") as f:
    f.create_dataset("data", data=X_down)

# FashionMNIST
fmnist = td.FashionMNIST("fashionMNIST", transform=transforms.ToTensor(), download=True)
data = DataLoader(dataset=mnist, batch_size=60000)
X = next(iter(data))[0].view(60000, -1).detach().cpu().numpy()

# centering
X = X - X.mean(0, keepdims=True)
with h5py.File("../datasets/fmnist784.h5", "w") as f:
    f.create_dataset("data", data=X)

# PCA
dims = 30
U, s, Vh = scipy.linalg.svd(X.transpose(), full_matrices=False)
U = U[:, :dims]
X_down = X @ U
with h5py.File("../datasets/fmnist30.h5", "w") as f:
    f.create_dataset("data", data=X_down)
