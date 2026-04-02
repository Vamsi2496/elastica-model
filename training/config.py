import torch

class Config:
    # ── Data ────────────────────────────────────────────────────────── #
    HDF5_PATH     = "data.h5"          # ← your actual filename
    N_NODES       = 201
    TRAIN_SPLIT   = 0.80
    VAL_SPLIT     = 0.10

    # ── HDF5 key mapping (matches your actual file) ──────────────────── #
    KEY_PHI1      = "phi1"
    KEY_PHI2      = "phi2"
    KEY_D         = "d"
    KEY_ARC       = "t"                     # arc length stored as "t"
    KEY_THETA     = "u1"                    # theta stored as "u1"
    KEY_PARAMS    = "parameters"            # shape (N, 9)

    # parameters array indices
    IDX_FX        = 0
    IDX_FY        = 1
    IDX_XTIP      = 2
    IDX_YTIP      = 3
    IDX_ASYMM     = 4
    IDX_SYMM      = 5
    IDX_M1        = 6                       # moment at left end
    IDX_M2        = 7                       # moment at right end

    # What goes into the scalar head target: Fx, Fy, M1, M2
    SCALAR_NAMES  = ["Fx", "Fy", "M_left", "M_right"]

    # ── Model ────────────────────────────────────────────────────────── #
    HIDDEN_DIM    = 512
    N_BLOCKS      = 6
    N_FREQ        = 64
    OMEGA         = 30.0

    # ── Loss weights ─────────────────────────────────────────────────── #
    LAMBDA_THETA  = 1.0
    LAMBDA_PHYS   = 1.0
    LAMBDA_CONS   = 1.0
    FX_WEIGHT     = 10.0 
    EI            = 1.0                     # ← set your actual EI value

    # ── Training ─────────────────────────────────────────────────────── #
    BATCH_SIZE    = 2048
    EPOCHS        = 60
    LR            = 3e-4
    WEIGHT_DECAY  = 1e-5
    NUM_WORKERS   = 8
    GRAD_CLIP     = 1.0
    MIXED_PREC    = True

    # ── Checkpointing ────────────────────────────────────────────────── #
    CKPT_DIR      = "checkpoints"
    CKPT_BEST     = "checkpoints/best_model.pt"
    NORM_STATS    = "norm_stats.npz"
    LOG_INTERVAL  = 50