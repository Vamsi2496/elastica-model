import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from config import Config


class ElasticaDataset(Dataset):
    def __init__(self, path: str, compute_stats: bool = True):
        self.path  = path
        self._file = None

        with h5py.File(path, "r") as f:
            self.N = f[Config.KEY_PHI1].shape[0]
            print(f"  Total samples in file : {self.N:,}")

            if compute_stats:
                print("Computing normalisation statistics …")

                X = np.stack([
                    f[Config.KEY_PHI1][:].astype(np.float32),
                    f[Config.KEY_PHI2][:].astype(np.float32),
                    f[Config.KEY_D][:].astype(np.float32),
                ], axis=1)
                self.x_mean = X.mean(0).astype(np.float32)
                self.x_std  = X.std(0).astype(np.float32) + 1e-8

                params = f[Config.KEY_PARAMS][:].astype(np.float32)
                Y = np.stack([
                    params[:, Config.IDX_FX],
                    params[:, Config.IDX_FY],
                    params[:, Config.IDX_M1],
                    params[:, Config.IDX_M2],
                ], axis=1)
                self.y_mean = Y.mean(0).astype(np.float32)
                self.y_std  = Y.std(0).astype(np.float32)  + 1e-8

                theta_all    = np.vstack(
                    f[Config.KEY_THETA][:]).astype(np.float32)
                self.t_mean  = float(theta_all.mean())
                self.t_std   = float(theta_all.std())       + 1e-8

                arc_all      = np.vstack(
                    f[Config.KEY_ARC][:]).astype(np.float32)
                self.arc_max = float(arc_all.max())

                np.savez(Config.NORM_STATS,
                         x_mean   = self.x_mean,
                         x_std    = self.x_std,
                         y_mean   = self.y_mean,
                         y_std    = self.y_std,
                         t_mean   = self.t_mean,
                         t_std    = self.t_std,
                         arc_max  = self.arc_max)

                print(f"  phi1/phi2/d mean : {self.x_mean}")
                print(f"  phi1/phi2/d std  : {self.x_std}")
                print(f"  Fx/Fy/M1/M2 mean : {self.y_mean}")
                print(f"  Fx/Fy/M1/M2 std  : {self.y_std}")
                print(f"  theta mean       : {self.t_mean:.6f}")
                print(f"  theta std        : {self.t_std:.6f}")
                print(f"  arc length max   : {self.arc_max:.6f}")
                print(f"  Stats saved → {Config.NORM_STATS}")

            else:
                st = np.load(Config.NORM_STATS)
                self.x_mean  = st["x_mean"]
                self.x_std   = st["x_std"]
                self.y_mean  = st["y_mean"]
                self.y_std   = st["y_std"]
                self.t_mean  = float(st["t_mean"])
                self.t_std   = float(st["t_std"])
                self.arc_max = float(st["arc_max"])
                print(f"  Norm stats loaded ← {Config.NORM_STATS}")

    def _get_file(self):
        if self._file is None:
            self._file = h5py.File(self.path, "r", swmr=True)
        return self._file

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        f = self._get_file()

        x = np.array([
            f[Config.KEY_PHI1][idx],
            f[Config.KEY_PHI2][idx],
            f[Config.KEY_D][idx],
        ], dtype=np.float32)
        x = (x - self.x_mean) / self.x_std

        params = np.array(f[Config.KEY_PARAMS][idx], dtype=np.float32)
        y = np.array([
            params[Config.IDX_FX],
            params[Config.IDX_FY],
            params[Config.IDX_M1],
            params[Config.IDX_M2],
        ], dtype=np.float32)
        y = (y - self.y_mean) / self.y_std

        arc   = np.array(f[Config.KEY_ARC][idx],   dtype=np.float32)
        arc   = arc / self.arc_max

        theta = np.array(f[Config.KEY_THETA][idx],  dtype=np.float32)
        theta = (theta - self.t_mean) / self.t_std

        return (torch.from_numpy(x),
                torch.from_numpy(y),
                torch.from_numpy(arc),
                torch.from_numpy(theta))


def get_loaders(path: str, compute_stats: bool = True):
    dataset = ElasticaDataset(path, compute_stats=compute_stats)
    N       = len(dataset)

    idx = np.arange(N)
    np.random.seed(42)
    np.random.shuffle(idx)

    n_train = int(Config.TRAIN_SPLIT * N)
    n_val   = int(Config.VAL_SPLIT   * N)

    splits = {
        "train": idx[:n_train],
        "val"  : idx[n_train : n_train + n_val],
        "test" : idx[n_train + n_val:]
    }

    print(f"  Train : {len(splits['train']):,}")
    print(f"  Val   : {len(splits['val']):,}")
    print(f"  Test  : {len(splits['test']):,}")

    def make_loader(indices, shuffle):
        return DataLoader(
            Subset(dataset, indices),
            batch_size         = Config.BATCH_SIZE,
            shuffle            = shuffle,
            num_workers        = Config.NUM_WORKERS,
            pin_memory         = Config.PIN_MEMORY,
            persistent_workers = Config.NUM_WORKERS > 0,
            prefetch_factor    = 2 if Config.NUM_WORKERS > 0 else None,
        )

    return (make_loader(splits["train"], shuffle=True),
            make_loader(splits["val"],   shuffle=False),
            make_loader(splits["test"],  shuffle=False),
            dataset)