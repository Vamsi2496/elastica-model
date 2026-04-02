import torch
import numpy as np
from model  import ElasticaScalarNet
from config import Config



class ElasticaPredictor:
    """
    Load model once at startup.
    Call predict() or predict_batch() as many times as needed.

    Usage:
        predictor = ElasticaPredictor()
        result    = predictor.predict(-36.58, 36.58, 0.9)
        results   = predictor.predict_batch(phi1_arr, phi2_arr, d_arr)
    """

    def __init__(self,
                 ckpt_path  = Config.CKPT_BEST,
                 norm_path  = Config.NORM_STATS):

        self.device = Config.DEVICE

        # Load stats
        stats        = np.load(norm_path)
        self.x_mean  = stats["x_mean"].astype(np.float32)
        self.x_std   = stats["x_std"].astype(np.float32)
        self.y_mean  = stats["y_mean"].astype(np.float32)
        self.y_std   = stats["y_std"].astype(np.float32)

        # Load model
        ckpt         = torch.load(ckpt_path, map_location=self.device)
        self.model   = ElasticaScalarNet().to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

        print(f"Model loaded — epoch {ckpt['epoch']} | "
              f"val_loss={ckpt['val_loss']:.6f}")

    def _normalise(self, phi1, phi2, d):
        x = np.array([phi1, phi2, d], dtype=np.float32)
        return (x - self.x_mean) / self.x_std

    def _denormalise(self, y_norm):
        return y_norm * self.y_std + self.y_mean

    # ── Single prediction ─────────────────────────────────────────── #
    def predict(self, phi1, phi2, d):
        x   = self._normalise(phi1, phi2, d)
        x_t = torch.from_numpy(x).unsqueeze(0).to(self.device)  # (1, 3)

        with torch.no_grad():
            y_norm = self.model(x_t)

        y = self._denormalise(y_norm.cpu().numpy()[0])

        return {
            "Fx"     : float(y[0]),
            "Fy"     : float(y[1]),
            "M_left" : float(y[2]),
            "M_right": float(y[3]),
        }

    # ── Batch prediction ──────────────────────────────────────────── #
    def predict_batch(self, phi1_arr, phi2_arr, d_arr,
                      batch_size=4096):
        """
        Handles arbitrarily large inputs by chunking into batches.
        Avoids OOM on large arrays.
        """
        phi1_arr = np.asarray(phi1_arr, dtype=np.float32)
        phi2_arr = np.asarray(phi2_arr, dtype=np.float32)
        d_arr    = np.asarray(d_arr,    dtype=np.float32)

        X = np.stack([phi1_arr, phi2_arr, d_arr], axis=1)  # (N, 3)
        X = (X - self.x_mean) / self.x_std

        results = []
        for i in range(0, len(X), batch_size):
            chunk = torch.from_numpy(X[i:i+batch_size]).to(self.device)
            with torch.no_grad():
                y_norm = self.model(chunk)
            results.append(
                self._denormalise(y_norm.cpu().numpy())
            )

        Y = np.concatenate(results, axis=0)                # (N, 4)

        return {
            "Fx"     : Y[:, 0],
            "Fy"     : Y[:, 1],
            "M_left" : Y[:, 2],
            "M_right": Y[:, 3],
        }


# ── Example usage ─────────────────────────────────────────────── #
if __name__ == "__main__":
    predictor = ElasticaPredictor()

    # Single
    r = predictor.predict(phi1=0, phi2=0, d=0.91)
    print(f"\nSingle prediction:")
    print(f"  Fx={r['Fx']:.4f}  Fy={r['Fy']:.6f}  "
          f"ML={r['M_left']:.4f}  MR={r['M_right']:.4f}")

    # Batch — 1000 random configurations
    N       = 1000
    phi1arr = np.random.uniform(-30,  30, N)
    phi2arr = np.random.uniform(-30,  30, N)
    darr    = np.random.uniform(0.60, 0.99, N)

    batch_r = predictor.predict_batch(phi1arr, phi2arr, darr)
    print(f"\nBatch prediction ({N} samples):")
    print(f"  Fx  : mean={batch_r['Fx'].mean():.4f}  "
          f"std={batch_r['Fx'].std():.4f}")
    print(f"  Fy  : mean={batch_r['Fy'].mean():.6f}  "
          f"std={batch_r['Fy'].std():.6f}")
    print(f"  ML  : mean={batch_r['M_left'].mean():.4f}  "
          f"std={batch_r['M_left'].std():.4f}")
    print(f"  MR  : mean={batch_r['M_right'].mean():.4f}  "
          f"std={batch_r['M_right'].std():.4f}")