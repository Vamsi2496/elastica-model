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

    SCALAR_NAMES = ["Energy", "Fx", "Fy", "M_left", "M_right"]

    # stored labels = SIGN * dU/dq
    SIGN_FX = -1.0
    SIGN_M1 = 1.0
    SIGN_M2 = 1.0

    # ── Model ─────────────────────────────────────────────────────────── #
    HIDDEN_DIM = 512
    N_BLOCKS = 8

    # ── Loss weights ──────────────────────────────────────────────────── #
    W_ENERGY_LABEL = 50.0
    W_ENERGY_THETA = 1.0
    W_SCALAR = 1.0
    W_LSTSQ = 1.0
    W_EQ = 1.0
    FX_WEIGHT = 10.0
    FY_WEIGHT = 0.5
    M_WEIGHT = 10.0
    EI = 1.0
    LAMBDA_STIFF = 0.2

    # ── Training ──────────────────────────────────────────────────────── #
    BATCH_SIZE = 32768
    EPOCHS = 25
    LR = 3e-4
    WEIGHT_DECAY = 1e-5
    GRAD_CLIP = 1.0
    LOG_INTERVAL = 20
    PATIENCE = 8
    MIN_DELTA = 1e-4

    # ── Device ────────────────────────────────────────────────────────── #
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
