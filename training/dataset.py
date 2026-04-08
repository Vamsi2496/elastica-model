import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from config import Config


class ElasticaDataset:
    def __init__(self, path: str, compute_stats: bool = True):
        self.path = path

        with h5py.File(path, "r") as f:
            self.N = f[Config.KEY_PHI1].shape[0]
            print(f" Total samples in file : {self.N:,}")
            print("Loading full dataset into memory …")

            phi1 = f[Config.KEY_PHI1][:].astype(np.float32)
            phi2 = f[Config.KEY_PHI2][:].astype(np.float32)
            d = f[Config.KEY_D][:].astype(np.float32)
            arc = np.vstack(f[Config.KEY_ARC][:]).astype(np.float32)
            theta = np.vstack(f[Config.KEY_THETA][:]).astype(np.float32)
            params = f[Config.KEY_PARAMS][:].astype(np.float32)

        X = np.stack([phi1, phi2, d], axis=1)
        Y = np.stack([
            params[:, Config.IDX_ENERGY],
            params[:, Config.IDX_FX],
            params[:, Config.IDX_FY],
            params[:, Config.IDX_M1],
            params[:, Config.IDX_M2],
        ], axis=1)

        if compute_stats:
            print("Computing normalisation statistics …")
            self.x_mean = X.mean(0).astype(np.float32)
            self.x_std = X.std(0).astype(np.float32) + 1e-8
            self.y_mean = Y.mean(0).astype(np.float32)
            self.y_std = Y.std(0).astype(np.float32) + 1e-8
            self.t_mean = float(theta.mean())
            self.t_std = float(theta.std()) + 1e-8
            self.arc_max = float(arc.max())

            np.savez(
                Config.NORM_STATS,
                x_mean=self.x_mean,
                x_std=self.x_std,
                y_mean=self.y_mean,
                y_std=self.y_std,
                t_mean=self.t_mean,
                t_std=self.t_std,
                arc_max=self.arc_max,
            )
            print(f" Stats saved → {Config.NORM_STATS}")
        else:
            st = np.load(Config.NORM_STATS)
            self.x_mean = st["x_mean"]
            self.x_std = st["x_std"]
            self.y_mean = st["y_mean"]
            self.y_std = st["y_std"]
            self.t_mean = float(st["t_mean"])
            self.t_std = float(st["t_std"])
            self.arc_max = float(st["arc_max"])
            print(f" Norm stats loaded ← {Config.NORM_STATS}")

        X = (X - self.x_mean) / self.x_std
        Y = (Y - self.y_mean) / self.y_std
        arc = arc / self.arc_max
        theta = (theta - self.t_mean) / self.t_std

        device = Config.DEVICE
        print(f"Moving full dataset to {device} …")
        self.x = torch.from_numpy(X).to(device)
        self.y = torch.from_numpy(Y).to(device)
        self.arc = torch.from_numpy(arc).to(device)
        self.theta = torch.from_numpy(theta).to(device)

        #mem_gb = (self.x.numel() + self.y.numel() + self.arc.numel() + self.theta.numel()) * 4 / 1e9
        #print(f" Dataset on device: {mem_gb:.2f} GB")


def get_loaders(path: str, compute_stats: bool = True):
    dataset = ElasticaDataset(path, compute_stats=compute_stats)
    N = dataset.N

    idx = np.arange(N)
    np.random.seed(42)
    np.random.shuffle(idx)

    n_train = int(Config.TRAIN_SPLIT * N)
    n_val = int(Config.VAL_SPLIT * N)

    train_idx = torch.from_numpy(idx[:n_train]).long().to(Config.DEVICE)
    val_idx = torch.from_numpy(idx[n_train:n_train + n_val]).long().to(Config.DEVICE)
    test_idx = torch.from_numpy(idx[n_train + n_val:]).long().to(Config.DEVICE)

    train_ds = TensorDataset(dataset.x[train_idx], dataset.y[train_idx], dataset.arc[train_idx], dataset.theta[train_idx])
    val_ds = TensorDataset(dataset.x[val_idx], dataset.y[val_idx], dataset.arc[val_idx], dataset.theta[val_idx])
    test_ds = TensorDataset(dataset.x[test_idx], dataset.y[test_idx], dataset.arc[test_idx], dataset.theta[test_idx])

    print(f" Train : {len(train_ds):,}")
    print(f" Val   : {len(val_ds):,}")
    print(f" Test  : {len(test_ds):,}")

    def make_loader(ds, shuffle):
        return DataLoader(ds, batch_size=Config.BATCH_SIZE, shuffle=shuffle, num_workers=0, pin_memory=False)

    return (
        make_loader(train_ds, shuffle=True),
        make_loader(val_ds, shuffle=False),
        make_loader(test_ds, shuffle=False),
        dataset,
    )
