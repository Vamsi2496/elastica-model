import torch


class Config:
    # ── Data ─────────────────────────────────────────────────────────── #
    HDF5_PATH = "auto_data.h5"
    N_NODES = 201
    TRAIN_SPLIT = 0.80
    VAL_SPLIT = 0.10

    # ── HDF5 keys ─────────────────────────────────────────────────────── #
    KEY_PHI1 = "phi1"
    KEY_PHI2 = "phi2"
    KEY_D = "d"
    KEY_ARC = "t"
    KEY_THETA = "u1"
    KEY_PARAMS = "parameters"

    IDX_FX = 0
    IDX_FY = 1
    IDX_M1 = 6
    IDX_M2 = 7
    IDX_ENERGY = 8

    SCALAR_NAMES = ["Energy", "Fx", "M_left", "M_right"]
    STIFFNESS_NAMES = ["phi1", "phi2", "d"]

    # Stored-label convention: label = SIGN * dU/dq
    SIGN_FX = -1.0
    SIGN_M1 = -1.0
    SIGN_M2 = -1.0

    # ── Model ─────────────────────────────────────────────────────────── #
    HIDDEN_DIM = 512
    N_BLOCKS = 6

    # ── Loss weights ──────────────────────────────────────────────────── #
    W_ENERGY = 1.0
    W_FX = 10.0
    W_M1 = 1.0
    W_M2 = 1.0

    # Optional smoothness penalty on Hessian magnitude
    LAMBDA_STIFF = 0.0

    EI = 1.0

    # ── Training ──────────────────────────────────────────────────────── #
    BATCH_SIZE = 32768
    EPOCHS = 50
    LR = 3e-4
    WEIGHT_DECAY = 1e-5
    GRAD_CLIP = 1.0
    LOG_INTERVAL = 20
    PATIENCE = 8
    MIN_DELTA = 1e-4

    # ── Device — auto detected ─────────────────────────────────────────── #
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    USE_GPU = torch.cuda.is_available()
    MIXED_PREC = False
    PIN_MEMORY = False
    NUM_WORKERS = 0
    COMPILE = False

    # ── Checkpointing ─────────────────────────────────────────────────── #
    CKPT_DIR = "checkpoints_energy"
    CKPT_BEST = "checkpoints_energy/best_model.pt"
    NORM_STATS = "norm_stats_energy.npz"
