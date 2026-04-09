import torch

class Config:
    # ── Data ─────────────────────────────────────────────────────────── #
    HDF5_PATH    = "auto_data.h5"
    N_NODES      = 201
    TRAIN_SPLIT  = 0.80
    VAL_SPLIT    = 0.10

    # ── HDF5 keys ─────────────────────────────────────────────────────── #
    KEY_PHI1     = "phi1"
    KEY_PHI2     = "phi2"
    KEY_D        = "d"
    KEY_ARC      = "t"
    KEY_THETA    = "u1"
    KEY_PARAMS   = "parameters"

    IDX_FX       = 0
    IDX_FY       = 1
    IDX_M1       = 6
    IDX_M2       = 7

    SCALAR_NAMES = ["Fx", "Fy", "M_left", "M_right"]

    # ── Model ─────────────────────────────────────────────────────────── #
    HIDDEN_DIM   = 512
    N_BLOCKS     = 6

    # ── Loss weights ──────────────────────────────────────────────────── #
    LAMBDA_PHYS  = 1.0
    LAMBDA_CONS  = 1.0
    FX_WEIGHT    = 10.0     # Fx has widest range — boost supervised signal
    FY_WEIGHT    = 0.5      # Fy near-zero in symmetric configs — reduce noise
    EI           = 1.0

    # ── Training ──────────────────────────────────────────────────────── #
    BATCH_SIZE   = 131072
    EPOCHS       = 50
    LR           = 3e-4
    WEIGHT_DECAY = 1e-5
    GRAD_CLIP    = 1.0
    LOG_INTERVAL = 20

    # ── Device — auto detected ─────────────────────────────────────────── #
    DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    USE_GPU      = torch.cuda.is_available()
    MIXED_PREC   = USE_GPU          # AMP only on GPU — float16 forward pass
    PIN_MEMORY   = False
    #PIN_MEMORY   = USE_GPU          # pinned memory only useful with GPU
    #NUM_WORKERS  = 14 if USE_GPU else 0
    NUM_WORKERS  = 0
    COMPILE      = USE_GPU          # torch.compile only on GPU (PyTorch >= 2.0)

    # ── Checkpointing ─────────────────────────────────────────────────── #
    CKPT_DIR     = "checkpoints"
    CKPT_BEST    = "checkpoints/best_model.pt"
    NORM_STATS   = "norm_stats.npz"