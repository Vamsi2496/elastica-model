import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from config import Config


def _compute_d_weights(d_phys: np.ndarray, n_bins: int) -> np.ndarray:
    d_min, d_max = d_phys.min(), d_phys.max()
    bin_idx = np.floor(
        (d_phys - d_min) / (d_max - d_min + 1e-8) * n_bins
    ).astype(int)
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    counts = np.bincount(bin_idx, minlength=n_bins).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    weights = 1.0 / counts[bin_idx]
    weights = weights / weights.mean()
    return weights.astype(np.float32)


class ElasticaDataset:
    def __init__(self, path: str, compute_stats: bool = True):
        self.path = path
        with h5py.File(path, "r") as f:
            total = f[Config.KEY_PHI1].shape[0]
            print(f"Total samples in file: {total:,}")
            phi1   = f[Config.KEY_PHI1][:].astype(np.float32)
            phi2   = f[Config.KEY_PHI2][:].astype(np.float32)
            d      = f[Config.KEY_D][:].astype(np.float32)
            params = f[Config.KEY_PARAMS][:].astype(np.float32)

        if Config.D_SLICE is not None:
            mask   = np.abs(d - Config.D_SLICE) <= Config.D_SLICE_TOL
            phi1   = phi1[mask]; phi2 = phi2[mask]; d = d[mask]
            params = params[mask]
            self.N = int(mask.sum())
            print(f"d-slice {Config.D_SLICE}±{Config.D_SLICE_TOL}: "
                  f"{self.N:,} samples ({100*self.N/total:.1f}% of total)")
        else:
            self.N = total
            print("Using full dataset (no d-slice)")

        self.d_phys_raw = d.copy()

        # Inputs: phi1, phi2 only — d excluded so Fx is not defined
        X = np.stack([phi1, phi2], axis=1)
        # Outputs: Energy, M_left, M_right
        Y = np.stack([params[:, Config.IDX_ENERGY],
                      params[:, Config.IDX_M1],
                      params[:, Config.IDX_M2]], axis=1)

        if compute_stats:
            print("Computing normalization statistics …")
            self.x_mean = X.mean(0).astype(np.float32)
            self.x_std  = X.std(0).astype(np.float32) + 1e-8
            self.y_mean = Y.mean(0).astype(np.float32)
            self.y_std  = Y.std(0).astype(np.float32) + 1e-8
            np.savez(Config.NORM_STATS,
                     x_mean=self.x_mean, x_std=self.x_std,
                     y_mean=self.y_mean, y_std=self.y_std)
            print(f"Stats saved → {Config.NORM_STATS}")
        else:
            st = np.load(Config.NORM_STATS)
            self.x_mean = st["x_mean"]
            self.x_std  = st["x_std"]
            self.y_mean = st["y_mean"]
            self.y_std  = st["y_std"]
            print(f"Norm stats loaded ← {Config.NORM_STATS}")

        X = (X - self.x_mean) / self.x_std
        Y = (Y - self.y_mean) / self.y_std

        device = Config.DEVICE
        print(f"Moving full dataset to {device} …")
        self.x = torch.from_numpy(X).to(device)
        self.y = torch.from_numpy(Y).to(device)


def get_loaders(path: str, compute_stats: bool = True):
    dataset = ElasticaDataset(path, compute_stats=compute_stats)
    N   = dataset.N
    idx = np.arange(N)
    np.random.seed(Config.RANDOM_SEED)
    np.random.shuffle(idx)
    n_train = int(Config.TRAIN_SPLIT * N)
    n_val   = int(Config.VAL_SPLIT   * N)

    train_idx = torch.from_numpy(idx[:n_train]).long().to(Config.DEVICE)
    val_idx   = torch.from_numpy(idx[n_train:n_train + n_val]).long().to(Config.DEVICE)
    test_idx  = torch.from_numpy(idx[n_train + n_val:]).long().to(Config.DEVICE)

    train_ds = TensorDataset(dataset.x[train_idx], dataset.y[train_idx])
    val_ds   = TensorDataset(dataset.x[val_idx],   dataset.y[val_idx])
    test_ds  = TensorDataset(dataset.x[test_idx],  dataset.y[test_idx])

    print(f"Train: {len(train_ds):,}")
    print(f"Val:   {len(val_ds):,}")
    print(f"Test:  {len(test_ds):,}")

    def make_loader(ds, shuffle):
        return DataLoader(ds, batch_size=Config.BATCH_SIZE, shuffle=shuffle,
                          num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY)

    if Config.WEIGHTED_D_SAMPLING:
        d_train_phys = dataset.d_phys_raw[idx[:n_train]]
        w = _compute_d_weights(d_train_phys, n_bins=Config.D_WEIGHT_BINS)
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(w),
            num_samples=len(w),
            replacement=True,
        )
        d_min = d_train_phys.min(); d_max = d_train_phys.max()
        print(f"Weighted sampler: d=[{d_min:.3f},{d_max:.3f}], "
              f"{Config.D_WEIGHT_BINS} bins, "
              f"max/min weight ratio = {w.max()/w.min():.1f}x")
        train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE,
                                  sampler=sampler,
                                  num_workers=Config.NUM_WORKERS,
                                  pin_memory=Config.PIN_MEMORY)
    else:
        train_loader = make_loader(train_ds, shuffle=True)

    val_loader  = make_loader(val_ds,  shuffle=False)
    test_loader = make_loader(test_ds, shuffle=False)
    return train_loader, val_loader, test_loader, dataset
