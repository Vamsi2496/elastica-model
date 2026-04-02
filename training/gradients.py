import torch
import numpy as np
from model  import ElasticaScalarNet
from config import Config


def load_model(device):
    ckpt  = torch.load(Config.CKPT_BEST, map_location=device)
    model = ElasticaScalarNet().to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def get_gradients(phi1, phi2, d,
                  norm_path = Config.NORM_STATS,
                  ckpt_path = Config.CKPT_BEST):
    """
    Compute gradients of (Fx, Fy, ML, MR) with respect to (phi1, phi2, d).

    Inputs  (physical units):
        phi1, phi2 : float  boundary angles in degrees
        d          : float  end distance

    Returns dict:
        "values"    : dict  predicted Fx, Fy, ML, MR
        "gradients" : dict  each entry is [dF/dphi1, dF/dphi2, dF/dd]
        "jacobian"  : np.ndarray  shape (4, 3) — full J[i,j] = dF_i/dx_j
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stats  = np.load(norm_path)

    x_mean = stats["x_mean"].astype(np.float32)  # (3,)
    x_std  = stats["x_std"].astype(np.float32)   # (3,)
    y_mean = stats["y_mean"].astype(np.float32)  # (4,)
    y_std  = stats["y_std"].astype(np.float32)   # (4,)

    # ── Normalise input ──────────────────────────────────────────── #
    phi_raw = np.array([phi1, phi2, d], dtype=np.float32)
    phi_norm = (phi_raw - x_mean) / x_std                  # (3,)

    phi_t = torch.tensor(phi_norm, dtype=torch.float32,
                         requires_grad=True,
                         device=device).unsqueeze(0)        # (1, 3)

    # ── Forward pass ─────────────────────────────────────────────── #
    model  = load_model(device)
    y_norm = model(phi_t)                                   # (1, 4) normalised

    # ── Denormalise predictions ───────────────────────────────────── #
    y_std_t  = torch.tensor(y_std,  device=device)
    y_mean_t = torch.tensor(y_mean, device=device)
    y_phys   = y_norm * y_std_t + y_mean_t                 # (1, 4) physical

    # ── Compute Jacobian: d(y_phys)/d(phi_norm) ──────────────────── #
    # Shape: (4, 3) — one row per output, one col per input
    J_norm = torch.zeros(4, 3, device=device)

    for i in range(4):
        grad = torch.autograd.grad(
            outputs      = y_phys[0, i],
            inputs       = phi_t,
            retain_graph = True,
        )[0]                                               # (1, 3)
        J_norm[i] = grad[0]

    # ── Chain rule: scale to physical units ──────────────────────── #
    # y_phys = y_norm * y_std + y_mean
    # x_phys = x_norm * x_std + x_mean
    #
    # d(y_phys_i)/d(x_phys_j) = J_norm[i,j] / x_std[j]
    # (y_std already absorbed into y_phys via denorm above)

    x_std_t = torch.tensor(x_std, device=device)          # (3,)
    J_phys  = J_norm / x_std_t.unsqueeze(0)               # (4, 3)
    J_phys  = J_phys.detach().cpu().numpy()

    output_names = Config.SCALAR_NAMES                     # Fx,Fy,ML,MR
    input_names  = ["phi1", "phi2", "d"]

    # ── Package results ───────────────────────────────────────────── #
    values = {name: float(y_phys[0, i].item())
              for i, name in enumerate(output_names)}

    gradients = {}
    for i, out_name in enumerate(output_names):
        gradients[out_name] = {
            inp_name: float(J_phys[i, j])
            for j, inp_name in enumerate(input_names)
        }

    return {
        "values"   : values,
        "gradients": gradients,
        "jacobian" : J_phys,               # (4, 3) numpy array
    }


def print_results(result):
    output_names = Config.SCALAR_NAMES
    input_names  = ["phi1", "phi2", "d"]

    print("\n── Predicted Values ─────────────────────────────────────")
    for name, val in result["values"].items():
        print(f"  {name:<10} = {val:.6f}")

    print("\n── Jacobian  d(output)/d(input) ─────────────────────────")
    print(f"  {'':12}", end="")
    for inp in input_names:
        print(f"  {inp:>12}", end="")
    print()
    print("  " + "-" * 52)

    for out_name in output_names:
        print(f"  {out_name:<12}", end="")
        for inp_name in input_names:
            val = result["gradients"][out_name][inp_name]
            print(f"  {val:>12.6f}", end="")
        print()

    print("\n── Sensitivity Summary ──────────────────────────────────")
    J = result["jacobian"]                                 # (4, 3)
    for i, out_name in enumerate(output_names):
        most_sensitive_idx = np.argmax(np.abs(J[i]))
        most_sensitive_inp = input_names[most_sensitive_idx]
        print(f"  {out_name:<10} most sensitive to  "
              f"{most_sensitive_inp}  "
              f"(|dF/dx| = {abs(J[i, most_sensitive_idx]):.4f})")


if __name__ == "__main__":
    # ── Example: one configuration ───────────────────────────────── #
    result = get_gradients(phi1=0, phi2=0, d=0.9)
    print_results(result)

    # ── Example: sweep over d, track dFx/dd ──────────────────────── #
    #print("\n── dFx/dd as d varies from 0.85 to 0.99 ────────────────")
#    print(f"  {'d':>6}  {'Fx':>10}  {'dFx/dd':>12}")
 #   print("  " + "-" * 32)
  #  for d_val in np.linspace(0.85, 0.99, 8):
   #     r = get_gradients(phi1=-36.58, phi2=36.58, d=float(d_val))
    #    print(f"  {d_val:>6.3f}  "
     #         f"{r['values']['Fx']:>10.4f}  "
      #        f"{r['gradients']['Fx']['d']:>12.6f}")