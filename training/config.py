import torch


class Config:
    HDF5_PATH = "auto_data_no_boundary.h5"
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

    # d-slice: set D_SLICE=None to use full dataset (recommended for weighted sampling)
    D_SLICE     = 0.85
    D_SLICE_TOL = 0.01

    # weighted sampling: equalise training exposure across the d distribution
    WEIGHTED_D_SAMPLING = False
    D_WEIGHT_BINS       = 40   # number of histogram bins for computing d-weights

    SIGN_FX = -1.0
    SIGN_M1 = -1.0
    SIGN_M2 = 1.0

    INPUT_DIM = 3   # phi1, phi2, d
    HIDDEN_LAYERS = [128, 128]

    # --- loss weights (proven best from commit history) ---
    W_ENERGY_LABEL = 1.0
    W_SCALAR       = 1.0
    FX_WEIGHT      = 10.0
    FY_WEIGHT      = 1.0
    M_WEIGHT       = 1.0
    FX_L4_WEIGHT   = 0.0
    EI             = 1.0
    W_ENERGY_THETA = 0.0
    LAMBDA_STIFF   = 0.0
    # --- curriculum: ramp from _INIT → target over CURRICULUM_EPOCHS ---
    CURRICULUM_EPOCHS = 20
    #W_ENERGY_LABEL_INIT = 50.0
    #M_WEIGHT_INIT = 1.0
    # --- loss schedule ---
    # Each entry: (config_attr, intro_epoch, ramp_epochs, init_value)
    LOSS_SCHEDULE = [
        # attr          intro  ramp  init
        #("W_ENERGY_LABEL", 1,   20,  50.0),   # 50→20 over first 20 epochs
        ("FX_WEIGHT",     300,    1,   0.0),   # 0→5 starting epoch 50
        ("M_WEIGHT",      5000,   1,   0.0),   # 0→10 starting epoch 50
        ("FY_WEIGHT",     5000,   1,   0.0),   # 0→10 starting epoch 50
    ]

    BATCH_SIZE    = 8192
    EPOCHS        = 3000
    LR            = 1e-3
    WEIGHT_DECAY  = 1e-5
    GRAD_CLIP     = 1.0
    LOG_INTERVAL  = 40
    PATIENCE      = 1200
    MIN_DELTA     = 1e-6
    LR_FACTOR     = 0.25
    LR_PATIENCE   = 25
    MIN_LR        = 1e-6
    LR_THRESHOLD  = 1e-4

    DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    USE_GPU   = torch.cuda.is_available()
    MIXED_PREC = False
    PIN_MEMORY = False
    NUM_WORKERS = 0
    COMPILE     = False

    CKPT_DIR   = "checkpoints_weighted"
    CKPT_BEST  = "checkpoints_weighted/best_model.pt"
    CKPT_LATEST= "checkpoints_weighted/latest_model.pt"
    NORM_STATS = "norm_stats_weighted.npz"
