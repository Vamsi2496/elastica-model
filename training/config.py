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

    SIGN_FX = -1.0
    SIGN_M1 = -1.0
    SIGN_M2 = 1.0

    INPUT_DIM = 3
    # --- plain MLP architecture (GELU, no Fourier, no residuals) ---
    HIDDEN_LAYERS = [128, 128]

    # --- loss weights (final / target values) ---
    W_ENERGY_LABEL = 1.0
    W_SCALAR = 1.0
    FX_WEIGHT = 1.0
    FY_WEIGHT = 1.0
    M_WEIGHT = 1.0
    FX_L4_WEIGHT = 0.0
    EI = 1.0
    W_ENERGY_THETA = 0.0
    LAMBDA_STIFF = 0.0

    # --- loss schedule ---
    # Each entry: (config_attr, intro_epoch, ramp_epochs, init_value)
    #   intro_epoch  — epoch at which the component is switched on (weight = 0 before this)
    #   ramp_epochs  — linearly ramp from init_value → target over this many epochs
    #                  (0 = step directly to target at intro_epoch)
    #   init_value   — value at intro_epoch (ramp start); ignored when ramp_epochs = 0
    # Target for each entry is the static weight defined above.
    # Components NOT listed here keep their static weight throughout training.
    LOSS_SCHEDULE = [
        # attr              intro  ramp  init
        ("FX_WEIGHT",      50,     1,    0.0),  # 0 → 5 over epochs 50-80
        ("M_WEIGHT",      300,     1,    0.0),  # 0 → 10 over epochs 100-120
        ("FY_WEIGHT",     590,     1,    0.0),  # 0 → 5 over epochs 50-80
    ]

    BATCH_SIZE = 32768
    EPOCHS = 600
    LR = 1e-3
    WEIGHT_DECAY = 1e-5
    GRAD_CLIP = 1.0
    LOG_INTERVAL = 40
    PATIENCE = 600
    MIN_DELTA = 1e-6
    LR_FACTOR = 0.5
    LR_PATIENCE = 6
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
    CKPT_LATEST = "checkpoints_energy/latest_model.pt"
    NORM_STATS = "norm_stats_energy.npz"
