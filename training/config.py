import torch


class Config:
    HDF5_PATH = "auto_data.h5"
    N_NODES = 201
    TRAIN_SPLIT = 0.80
    VAL_SPLIT = 0.10
    RANDOM_SEED = 42

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

    SIGN_FX = -1.0
    SIGN_M1 = 1.0
    SIGN_M2 = -1.0

    INPUT_DIM = 3
    HIDDEN_LAYERS = [512, 512]
    ACTIVATION = "gelu"
    USE_LAYER_NORM = False
    DROPOUT = 0.0

    W_ENERGY_LABEL = 50.0
    W_SCALAR = 1.0
    FX_WEIGHT = 2.0
    FY_WEIGHT = 0.0
    M_WEIGHT = 2.0
    EI = 1.0
    W_ENERGY_THETA = 0.0
    LAMBDA_STIFF = 0.0

    BATCH_SIZE = 8192
    EPOCHS = 15
    LR = 1e-3
    WEIGHT_DECAY = 1e-5
    GRAD_CLIP = 1.0
    LOG_INTERVAL = 20
    PATIENCE = 10
    MIN_DELTA = 1e-4
    LR_FACTOR = 0.5
    LR_PATIENCE = 4
    MIN_LR = 1e-6
    LR_THRESHOLD = 1e-4

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    USE_GPU = torch.cuda.is_available()
    MIXED_PREC = False
    PIN_MEMORY = False
    NUM_WORKERS = 0
    COMPILE = False

    CKPT_DIR = "checkpoints_energy"
    CKPT_BEST = "checkpoints_energy/best_model.pt"
    NORM_STATS = "norm_stats_energy.npz"
