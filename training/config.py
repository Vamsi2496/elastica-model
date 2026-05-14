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
    KEY_ARC    = "t"
    KEY_THETA  = "u1"
    KEY_PARAMS = "parameters"

    IDX_FX     = 0
    IDX_FY     = 1
    IDX_M1     = 6
    IDX_M2     = 7
    IDX_ENERGY = 8

    SCALAR_NAMES = ["Energy", "Fx", "Fy", "M_left", "M_right"]

    # full dataset
    D_SLICE     = None
    D_SLICE_TOL = 1e-8

    WEIGHTED_D_SAMPLING = False
    D_WEIGHT_BINS       = 40

    SIGN_FX = -1.0
    SIGN_M1 = -1.0
    SIGN_M2 =  1.0

    INPUT_DIM = 3   # phi1, phi2, d

    USE_FOURIER       = False
    FOURIER_FEATURES  = 128
    FOURIER_SIGMA_PHI = 1.5
    FOURIER_SIGMA_D   = 5.0

    HIDDEN_LAYERS = [128, 128]
    USE_RESIDUAL  = False

    # ---------------------------------------------------------------
    # NNP-inspired loss weights
    #
    # Neural Network Potential practice (Behler-Parrinello, DeePMD):
    #   Phase 1 — energy only:   establish the energy surface shape
    #   Phase 2 — forces heavy:  force gradients to be physically correct
    #
    # Old targets: W_ENERGY=20, FX=5,  M=10
    #              → energy ≈ forces in contribution (roughly balanced)
    #
    # New targets: W_ENERGY=5,  FX=30, M=50
    #              → forces 10-30x dominant post-curriculum
    # ---------------------------------------------------------------
    W_ENERGY_LABEL = 5.0    # reduced: energy surface needs shape, not dominance
    W_SCALAR       = 1.0
    FX_WEIGHT      = 30.0   # high: Fx is hardest gradient to fit, needs strong signal
    FY_WEIGHT      = 0.0    # disabled: Fy=(MR-ML)/d is derived, conflicts with M
    M_WEIGHT       = 50.0   # high: moments dominate gradient learning post-curriculum
    FX_L4_WEIGHT   = 0.0
    EI             = 1.0
    W_ENERGY_THETA = 0.0
    LAMBDA_STIFF   = 0.0

    # Schedule:
    #   W_ENERGY_LABEL : 50 → 5  over epochs  1–50  (slow ramp, stabilise surface)
    #   FX_WEIGHT      :  0 → 30 from epoch 50, over 30 epochs  (heavy Fx supervision)
    #   M_WEIGHT       :  0 → 50 from epoch 50, over 30 epochs  (heavy moment supervision)
    LOSS_SCHEDULE = [
        # attr              intro  ramp  init
        ("W_ENERGY_LABEL",    1,   50,  50.0),  # 50 → 5  over epochs 1–50
        ("FX_WEIGHT",        50,   30,   0.0),  # 0  → 30 over epochs 50–80
        ("M_WEIGHT",         50,   30,   0.0),  # 0  → 50 over epochs 50–80
    ]

    BATCH_SIZE    = 8192
    EPOCHS        = 2000
    LR            = 1e-3
    WEIGHT_DECAY  = 1e-5
    GRAD_CLIP     = 1.0
    LOG_INTERVAL  = 40
    PATIENCE      = 2000
    MIN_DELTA     = 1e-6
    LR_FACTOR     = 0.25
    LR_PATIENCE   = 10
    MIN_LR        = 1e-6
    LR_THRESHOLD  = 1e-4

    DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    USE_GPU     = torch.cuda.is_available()
    MIXED_PREC  = False
    PIN_MEMORY  = False
    NUM_WORKERS = 0
    COMPILE     = False

    CKPT_DIR    = "checkpoints_nnp"
    CKPT_BEST   = "checkpoints_nnp/best_model.pt"
    CKPT_LATEST = "checkpoints_nnp/latest_model.pt"
    NORM_STATS  = "norm_stats_nnp.npz"
