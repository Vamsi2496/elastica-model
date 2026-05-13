import torch


class Config:
    HDF5_PATH = "auto_data_no_boundary.h5"
    N_NODES = 201
    TRAIN_SPLIT = 0.80
    VAL_SPLIT = 0.10
    RANDOM_SEED = 42

    KEY_PHI1   = "phi1"
    KEY_PHI2   = "phi2"
    KEY_D      = "d"
    KEY_PARAMS = "parameters"

    IDX_M1     = 6
    IDX_M2     = 7
    IDX_ENERGY = 8

    # Outputs: Energy, M_left, M_right only. Fx is not an input gradient here.
    SCALAR_NAMES = ["Energy", "M_left", "M_right"]

    # --- d-slice ---
    D_SLICE     = 0.65
    D_SLICE_TOL = 1e-8

    WEIGHTED_D_SAMPLING = False
    D_WEIGHT_BINS       = 40

    SIGN_M1 = -1.0
    SIGN_M2 =  1.0

    # 2 inputs: phi1, phi2 only
    INPUT_DIM = 2

    # --- Architecture: plain MLP, no Fourier ---
    USE_FOURIER       = False
    FOURIER_FEATURES  = 128
    FOURIER_SIGMA_PHI = 1.5
    FOURIER_SIGMA_D   = 2.5

    HIDDEN_LAYERS = [128, 128]
    USE_RESIDUAL  = False

    # --- loss weights ---
    W_ENERGY_LABEL = 1.0
    W_SCALAR       = 1.0
    M_WEIGHT       = 3.0
    EI             = 1.0
    

    # --- loss schedule ---
    LOSS_SCHEDULE = [
        # attr              intro  ramp  init
        #("W_ENERGY_LABEL",    1,   20,  50.0),
        ("M_WEIGHT",         300,   1,   0.0),
    ]

    BATCH_SIZE   = 8192
    EPOCHS       = 3000
    LR           = 1e-3
    WEIGHT_DECAY = 1e-5
    GRAD_CLIP    = 1.0
    LOG_INTERVAL = 40
    PATIENCE     = 2000
    MIN_DELTA    = 1e-6
    LR_FACTOR    = 0.25
    LR_PATIENCE  = 15
    MIN_LR       = 1e-6
    LR_THRESHOLD = 1e-4

    DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    USE_GPU    = torch.cuda.is_available()
    PIN_MEMORY = False
    NUM_WORKERS = 0

    CKPT_DIR    = "checkpoints_slice_mlp"
    CKPT_BEST   = "checkpoints_slice_mlp/best_model.pt"
    CKPT_LATEST = "checkpoints_slice_mlp/latest_model.pt"
    NORM_STATS  = "norm_stats_slice_mlp.npz"
