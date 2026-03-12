import os
import json
import torch
import numpy as np
import sys

# Import from train.py
from train import run_dpfedgd_training, parse_args

# --- Configuration ---
# Dataset / backbone for this sweep
SWEEP_DATASET = "pathmnist"  # options: pathmnist, bloodmnist, dermamnist, cifar10, cifar10_binary, chestmnist
SWEEP_BACKBONE = "cifar100_resnet20"  # matches the CIFAR-style pipeline

# If you use chestmnist, you likely want the medical backbone; otherwise keep CIFAR-style
CHESTMNIST_BACKBONE = "medical_resnet18"

epsilons = [0.5]
rounds_list = [1, 2, 5, 10, 20, 30, 40, 50, 60, 70]

# SOFIM configuration - SET THIS TO ENABLE/DISABLE SOFIM
# ALGORITHM SELECTION
USE_FEDNEW_FC = False  # Set to True to sweep FedNew-FC
USE_FEDFC = False      # Set to True to sweep FedFC (Algorithm 3)
USE_SOFIM = True       # Set to True to sweep SOFIM
USE_SCAFFOLD = False   # Set to True to sweep DP-SCAFFOLD
NO_DP_BASELINE = False  # set True if you want no-DP runs

if sum([USE_FEDNEW_FC, USE_FEDFC, USE_SOFIM, USE_SCAFFOLD]) > 1:
    raise ValueError("Enable only one of USE_FEDNEW_FC / USE_FEDFC / USE_SOFIM / USE_SCAFFOLD")

# FedNew Hyperparameters
FN_ALPHA = 0.1
FN_RHO = 0.1
FN_C1 = 1.0
FN_C2 = 1.0
FN_C_PRIMAL = 1.0
SOFIM_BETA = 0.9
SOFIM_RHO = 0.5

# FedFC Hyperparameters (Algorithm 3)
FC_CC = 1.0       # feature clipping C_c
FC_SIGMA_C = 1.0  # covariance noise sigma_c (ignored if NO_DP_BASELINE=True)
FC_GAMMA = 1.0    # covariance regularization gamma

# Output directory
results_dir = "./results_sweep"
os.makedirs(results_dir, exist_ok=True)

if USE_FEDNEW_FC:
    algorithm_name = "FedNew-FC"
elif USE_FEDFC:
    algorithm_name = "FedFC"
elif USE_SOFIM:
    algorithm_name = "SOFIM"
elif USE_SCAFFOLD:
    algorithm_name = "SCAFFOLD"
else:
    algorithm_name = "FedGD"

print(f"Starting DP-{algorithm_name} Sweep: {len(epsilons)} epsilons x {len(rounds_list)} round configs")
if USE_SOFIM:
    print(f"SOFIM enabled: β={SOFIM_BETA}, ρ={SOFIM_RHO}")

# 1. Get default args structure from train.py
sys.argv = ['train.py']
default_args = parse_args()

for eps in epsilons:
    for rounds in rounds_list:
        # Dynamic filename based on algorithm
        if USE_FEDNEW_FC:
            algorithm_name = "FedNew-FC"
            filename = f"fednew_eps{eps}_rounds{rounds}.json"
        elif USE_FEDFC:
            algorithm_name = "FedFC"
            filename = f"fedfc_eps{eps}_rounds{rounds}.json"
        elif USE_SOFIM:
            algorithm_name = "SOFIM"
            filename = f"sofim_eps{eps}_rounds{rounds}.json"
        elif USE_SCAFFOLD:
            algorithm_name = "SCAFFOLD"
            filename = f"scaffold_eps{eps}_rounds{rounds}.json"
        else:
            algorithm_name = "FedGD"
            filename = f"fedgd_eps{eps}_rounds{rounds}.json"

        filepath = os.path.join(results_dir, filename)

        print(f"\n▶️  Training DP-{algorithm_name}: ε={eps}, rounds={rounds} ...")

        # 2. Reset args to defaults to ensure clean state for every run
        import copy

        args = copy.deepcopy(default_args)

        # 3. Override with experiment settings
        args.use_fednew_fc = USE_FEDNEW_FC
        args.fn_alpha = FN_ALPHA
        args.fn_rho = FN_RHO
        args.fn_c1 = FN_C1
        args.fn_c2 = FN_C2
        args.fn_c_primal = FN_C_PRIMAL

        # FedFC settings (Algorithm 3)
        args.use_fedfc = USE_FEDFC
        if USE_FEDFC:
            args.fc_cc = FC_CC
            args.fc_gamma = FC_GAMMA
            # sigma_c is only required when DP is enabled
            args.fc_sigma_c = None if NO_DP_BASELINE else FC_SIGMA_C

        # DP-SCAFFOLD settings
        args.use_scaffold = USE_SCAFFOLD
        if USE_SCAFFOLD:
            # Keep K=1 by default to match the current privacy calibration
            args.scaffold_local_steps = 1
            args.scaffold_client_lr = 0.1
            args.scaffold_server_lr = 1.0

        args.epsilon = float(eps)
        args.delta = 1e-5
        args.federated_rounds = int(rounds)
        args.dataset = SWEEP_DATASET
        args.binary = (args.dataset == "cifar10_binary")

        # Select backbone based on dataset
        if args.dataset == "chestmnist":
            args.backbone = CHESTMNIST_BACKBONE
        else:
            args.backbone = SWEEP_BACKBONE

        # Fixed settings for consistency
        args.num_clients = 20
        args.clients_per_round = 20
        args.local_iterations = 1
        args.batch_size = 64
        args.learning_rate = 0.1
        args.gradient_clip_norm = 10.0
        args.sofim_disable_bias_correction = False
        args.sofim_adaptive_params = False
        args.sofim_warmup_rounds = 0

        # SOFIM settings (always set these, even if False)
        args.use_sofim = USE_SOFIM
        args.sofim_beta = SOFIM_BETA
        args.sofim_rho = SOFIM_RHO

        # Evaluation settings
        args.eval_every = 10
        args.seed = 42
        args.no_dp = NO_DP_BASELINE

        # 4. Run Training
        try:
            # Force clean memory before run
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            results = run_dpfedgd_training(args)

            # 5. Save specific result file
            if results is not None:
                # Add metadata about the run
                results['sweep_config'] = {
                    'epsilon': eps,
                    'rounds': rounds,
                    'algorithm': algorithm_name,
                    'use_fedfc': USE_FEDFC,
                    'use_sofim': USE_SOFIM,
                    'use_scaffold': USE_SCAFFOLD,
                }

                if USE_SOFIM:
                    results['sweep_config']['sofim_beta'] = SOFIM_BETA
                    results['sweep_config']['sofim_rho'] = SOFIM_RHO

                if USE_FEDFC:
                    results['sweep_config']['fc_cc'] = FC_CC
                    results['sweep_config']['fc_gamma'] = FC_GAMMA
                    results['sweep_config']['fc_sigma_c'] = (None if NO_DP_BASELINE else FC_SIGMA_C)

                if USE_SCAFFOLD:
                    results['sweep_config']['scaffold_local_steps'] = args.scaffold_local_steps
                    results['sweep_config']['scaffold_client_lr'] = args.scaffold_client_lr
                    results['sweep_config']['scaffold_server_lr'] = args.scaffold_server_lr

                with open(filepath, "w") as f:
                    json.dump(results, f, indent=2)
                print(f"✅ Saved: {filename}")
            else:
                print(f"❌ Failed: {filename} (results is None)")

        except Exception as e:
            print(f"❗ Error running {filename}: {e}")
            import traceback

            traceback.print_exc()

print(f"\n🎉 All DP-{algorithm_name} experiments completed.")