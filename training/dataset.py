import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from config import Config


class ElasticaDataset(Dataset):
    """
    Reads from auto_data.h5 with structure:
        phi1, phi2, d       : (N,)       float64  — scalar inputs
        t                   : (N,)       object   — each element is (201,) arc length
        u1                  : (N,)       object   — each element is (201,) theta
        parameters          : (N, 9)     float64  — [Fx,Fy,x_tip,y_tip,Asym,Sym,M1,M2,E]
        inflection_points   : (N,)       int32    — number of inflection points
    """

    def __init__(self, path: str, compute_stats: bool = True):
        self.path  = path
        self._file = None

        with h5py.File(path, "r") as f:
            self.N = f[Config.KEY_PHI1].shape[0]
            print(f"  Total samples in file : {self.N:,}")

            if compute_stats:
                print("Computing normalisation statistics …")

                # ── Input stats ──────────────────────────────────────── #
                phi1 = f[Config.KEY_PHI1][:].astype(np.float32)   # (N,)
                phi2 = f[Config.KEY_PHI2][:].astype(np.float32)   # (N,)
                d    = f[Config.KEY_D][:].astype(np.float32)       # (N,)
                X    = np.stack([phi1, phi2, d], axis=1)           # (N, 3)
                self.x_mean = X.mean(0).astype(np.float32)
                self.x_std  = X.std(0).astype(np.float32) + 1e-8

                # ── Scalar output stats ──────────────────────────────── #
                # parameters is a normal (N, 9) float array — no object issue
                params = f[Config.KEY_PARAMS][:].astype(np.float32)  # (N, 9)
                Y = np.stack([
                    params[:, Config.IDX_FX],
                    params[:, Config.IDX_FY],
                    params[:, Config.IDX_M1],
                    params[:, Config.IDX_M2],
                ], axis=1)                                            # (N, 4)
                self.y_mean = Y.mean(0).astype(np.float32)
                self.y_std  = Y.std(0).astype(np.float32) + 1e-8

                # ── Theta stats ──────────────────────────────────────── #
                # f["u1"][:] → (N,) dtype=object, each element is (201,)
                # np.vstack  → (N, 201) float array
                theta_all   = np.vstack(
                    f[Config.KEY_THETA][:]
                ).astype(np.float32)                                  # (N, 201)
                self.t_mean = float(theta_all.mean())
                self.t_std  = float(theta_all.std()) + 1e-8

                # ── Arc length stats ─────────────────────────────────── #
                # Same object array issue — use np.vstack
                arc_all      = np.vstack(
                    f[Config.KEY_ARC][:]
                ).astype(np.float32)                                  # (N, 201)
                self.arc_max = float(arc_all.max())

                # ── Save stats ───────────────────────────────────────── #
                np.savez(Config.NORM_STATS,
                         x_mean  = self.x_mean,
                         x_std   = self.x_std,
                         y_mean  = self.y_mean,
                         y_std   = self.y_std,
                         t_mean  = self.t_mean,
                         t_std   = self.t_std,
                         arc_max = self.arc_max)

                print(f"  phi1/phi2/d mean : {self.x_mean}")
                print(f"  phi1/phi2/d std  : {self.x_std}")
                print(f"  Fx/Fy/M1/M2 mean : {self.y_mean}")
                print(f"  Fx/Fy/M1/M2 std  : {self.y_std}")
                print(f"  theta mean       : {self.t_mean:.6f}")
                print(f"  theta std        : {self.t_std:.6f}")
                print(f"  arc length max   : {self.arc_max:.6f}")
                print(f"  Stats saved → {Config.NORM_STATS}")

            else:
                # Load pre-computed stats (used by test.py)
                st = np.load(Config.NORM_STATS)
                self.x_mean  = st["x_mean"]
                self.x_std   = st["x_std"]
                self.y_mean  = st["y_mean"]
                self.y_std   = st["y_std"]
                self.t_mean  = float(st["t_mean"])
                self.t_std   = float(st["t_std"])
                self.arc_max = float(st["arc_max"])
                print(f"  Norm stats loaded ← {Config.NORM_STATS}")

    # ------------------------------------------------------------------ #
    def _get_file(self):
        """
        Open HDF5 file lazily per worker.
        Avoids h5py multiprocessing conflicts with num_workers > 0.
        """
        if self._file is None:
            self._file = h5py.File(self.path, "r", swmr=True)
        return self._file

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        f = self._get_file()

        # ── Inputs (3,) ─────────────────────────────────────────────── #
        x = np.array([
            f[Config.KEY_PHI1][idx],
            f[Config.KEY_PHI2][idx],
            f[Config.KEY_D][idx],
        ], dtype=np.float32)
        x = (x - self.x_mean) / self.x_std                # normalised

        # ── Scalar targets (4,) from parameters array ───────────────── #
        # parameters is a normal float array — direct index works fine
        params = np.array(f[Config.KEY_PARAMS][idx], dtype=np.float32)  # (9,)
        y = np.array([
            params[Config.IDX_FX],
            params[Config.IDX_FY],
            params[Config.IDX_M1],
            params[Config.IDX_M2],
        ], dtype=np.float32)                               # (4,)
        y = (y - self.y_mean) / self.y_std                 # normalised

        # ── Arc length (201,) ────────────────────────────────────────── #
        # f["t"][idx] → python object wrapping a (201,) numpy array
        # np.array()  → converts it cleanly to float32 (201,)
        arc = np.array(f[Config.KEY_ARC][idx], dtype=np.float32)        # (201,)
        arc = arc / self.arc_max                           # normalised [0, 1]

        # ── Theta (201,) ─────────────────────────────────────────────── #
        # Same object-array unwrapping needed for "u1"
        theta = np.array(f[Config.KEY_THETA][idx], dtype=np.float32)    # (201,)
        theta = (theta - self.t_mean) / self.t_std         # normalised

        return (torch.from_numpy(x),
                torch.from_numpy(y),
                torch.from_numpy(arc),
                torch.from_numpy(theta))


# ------------------------------------------------------------------ #
def get_loaders(path: str, compute_stats: bool = True):
    """
    Returns train / val / test DataLoaders with 80 / 10 / 10 split.
    Split is deterministic (seed=42) so train/test sets are consistent
    between train.py and test.py runs.
    """
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

    print(f"  Train : {len(splits['train']):,} samples")
    print(f"  Val   : {len(splits['val']):,} samples")
    print(f"  Test  : {len(splits['test']):,} samples")

    def make_loader(indices, shuffle):
        return DataLoader(
            Subset(dataset, indices),
            batch_size         = Config.BATCH_SIZE,
            shuffle            = shuffle,
            num_workers        = Config.NUM_WORKERS,
            pin_memory         = True,
            persistent_workers = Config.NUM_WORKERS > 0,
            prefetch_factor    = 2 if Config.NUM_WORKERS > 0 else None,
        )

    return (make_loader(splits["train"], shuffle=True),
            make_loader(splits["val"],   shuffle=False),
            make_loader(splits["test"],  shuffle=False),
            dataset)