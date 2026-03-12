import argparse
import sys
import os
import torch
import json
import numpy as np
import time
from datetime import datetime
from typing import Dict, Any, Optional
import warnings

# Import the unified gradient-based training function
from train import run_dpfedgd_training


def parse_args():
    p = argparse.ArgumentParser(
        description="Gradient-Based Federated DP Training (DP-SOFIM / FedFC / FedGD) on CIFAR & MedMNIST",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="""
Example usage:
  # ResNet-18 IID training:
  python main.py --backbone resnet18 --num_clients 20 --clients_per_round 10 --epsilon 3.0

  # ResNet-50 with 224×224 inputs (ImageNet stem):
  python main.py --backbone resnet50 --resize_to_224 --num_clients 50 --epsilon 5.0

  # ResNet-50 with 32×32 inputs (CIFAR stem):
  python main.py --backbone resnet50 --no_resize_to_224 --epsilon 5.0

  # High heterogeneity non-IID:
  python main.py --backbone resnet50 --partition_type dirichlet --dirichlet_alpha 0.1 --user_level_dp --epsilon 8.0

  # Binary classification:
  python main.py --backbone resnet18 --binary --num_clients 15 --clients_per_round 8 --epsilon 4.0

  # MedMNIST (PathMNIST) with CIFAR-pretrained ResNet-20:
  python main.py --dataset pathmnist --backbone cifar100_resnet20 --num_clients 20 --clients_per_round 10 --epsilon 3.0

  # MedMNIST (BloodMNIST) with CIFAR-pretrained ResNet-20:
  python main.py --dataset bloodmnist --backbone cifar100_resnet20 --num_clients 20 --clients_per_round 10 --epsilon 3.0
        """
    )

    # === Model & Data Configuration ===
    model_group = p.add_argument_group('Model & Data')
    model_group.add_argument(
        "--backbone", type=str, default="resnet18",
        choices=["medical_resnet18","resnet18", "resnet50", "resnet20", "resnet32", "resnet44", "resnet56","cifar100_resnet20", "cifar100_resnet56"],
        help="Frozen feature extractor backbone"
    )
    model_group.add_argument(
        "--resize_to_224", action="store_true",
        help="For ResNet-50: use 224×224 ImageNet-style preprocessing"
    )
    model_group.add_argument(
        "--no_resize_to_224", action="store_true",
        help="For ResNet-50: use 32×32 CIFAR-style preprocessing"
    )
    model_group.add_argument(
        "--data_dir", type=str, default="./data",
        help="Directory for CIFAR-10 data"
    )
    model_group.add_argument(
        "--binary", action="store_true",
        help="Binary classification (classes 0 vs 1) instead of 10-way"
    )
    model_group.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        choices=[
            "cifar10",
            "cifar10_binary",
            "chestmnist",
            "pathmnist",
            "bloodmnist",
            "dermamnist",
        ],
        help="Dataset to use for federated training."
    )

    # === Federated Learning Setup ===
    fed_group = p.add_argument_group('Federated Learning')
    fed_group.add_argument(
        "--num_clients", type=int, default=20,
        help="Total number of federated clients"
    )
    fed_group.add_argument(
        "--clients_per_round", type=int, default=10,
        help="Clients participating per communication round"
    )
    fed_group.add_argument(
        "--federated_rounds", type=int, default=30,
        help="Number of federated communication rounds"
    )
    fed_group.add_argument(
        "--local_iterations", type=int, default=1,
        help="Local gradient computations per client per round"
    )
    fed_group.add_argument(
        "--client_sampling", type=str, default="random",
        choices=["random", "round_robin"],
        help="Client selection strategy"
    )

    # === Data Partitioning ===
    partition_group = p.add_argument_group('Data Partitioning')
    partition_group.add_argument(
        "--partition_type", type=str, default="iid",
        choices=["iid", "non_iid_classes", "dirichlet"],
        help="Data distribution strategy across clients"
    )
    partition_group.add_argument(
        "--classes_per_client", type=int, default=2,
        help="Classes per client for non_iid_classes partition"
    )
    partition_group.add_argument(
        "--dirichlet_alpha", type=float, default=0.5,
        help="Dirichlet concentration (lower = more heterogeneous)"
    )

    # === Training Hyperparameters ===
    train_group = p.add_argument_group('Training')
    train_group.add_argument(
        "--batch_size", type=int, default=64,
        help="Local batch size per client"
    )
    train_group.add_argument(
        "--client_lr", type=float, default=0.01,
        help="Client learning rate for local SGD"
    )
    train_group.add_argument(
        "--server_lr", type=float, default=1.0,
        help="Server learning rate for global updates"
    )
    train_group.add_argument(
        "--feature_clip_norm", type=float, default=1.0,
        help="L2 clipping norm for features (Algorithm 2 Step 1)"
    )
    train_group.add_argument(
        "--gradient_clip_norm", type=float, default=1.0,
        help="L2 clipping norm for gradients"
    )

    # === Gradient-Based DP-SOFIM Parameters ===
    # SOFIM parameters (server-side second-order optimization)
    p.add_argument("--use_sofim", action="store_true",
                   help="Use SOFIM (second-order) server optimizer instead of SGD")
    p.add_argument("--sofim_beta", type=float, default=0.9,
                   help="SOFIM momentum parameter β ∈ [0, 1)")
    p.add_argument("--sofim_rho", type=float, default=0.5,
                   help="SOFIM FIM regularization ρ > 0 (try 0.1, 0.5, or 1.0)")

    # === DP-FedNew-FC Parameters ===
    p.add_argument("--use_fednew_fc", action="store_true",
                   help="Use DP-FedNew-FC (Second-order ADMM)")
    p.add_argument("--fn_alpha", type=float, default=0.1,
                   help="FedNew dual step size (alpha)")
    p.add_argument("--fn_rho", type=float, default=0.1,
                   help="FedNew primal penalty (rho)")
    p.add_argument("--fn_c1", type=float, default=1.0,
                   help="Gradient clipping radius (C1)")
    p.add_argument("--fn_c2", type=float, default=1.0,
                   help="Auxiliary gradient clipping radius (C2)")
    p.add_argument("--fn_c_primal", type=float, default=1.0,
                   help="Primal variable clipping radius (C)")

    # === DP-FedFC Parameters (Algorithm 3) ===
    p.add_argument("--use_fedfc", action="store_true",
                   help="Use DP-FedFC (Algorithm 3: noisy covariance preconditioning + DP-FedGD updates)")
    p.add_argument("--fc_cc", type=float, default=1.0,
                   help="FedFC feature clipping radius C_c for covariance computation")
    p.add_argument("--fc_sigma_c", type=float, default=None,
                   help="FedFC covariance noise multiplier sigma_c (required for --use_fedfc unless --no_dp)")
    p.add_argument("--fc_gamma", type=float, default=1.0,
                   help="FedFC covariance regularization gamma added to C before inversion")

    # === DP-SCAFFOLD Parameters ===
    p.add_argument("--use_scaffold", action="store_true",
                   help="Use DP-SCAFFOLD (control variates)")
    p.add_argument("--scaffold_local_steps", type=int, default=5,
                   help="Number of local steps M for DP-SCAFFOLD")
    p.add_argument("--scaffold_client_lr", type=float, default=0.1,
                   help="Local step size η_l for DP-SCAFFOLD")
    p.add_argument("--scaffold_server_lr", type=float, default=None,
                   help="Server step size η_g for DP-SCAFFOLD (defaults to train.py learning_rate)")
    # === Differential Privacy ===
    privacy_group = p.add_argument_group('Differential Privacy')
    privacy_group.add_argument(
        "--epsilon", type=float, default=5.0,
        help="Target privacy parameter ε"
    )
    privacy_group.add_argument(
        "--delta", type=float, default=1e-5,
        help="Target privacy parameter δ"
    )
    privacy_group.add_argument(
        "--user_level_dp", action="store_true",
        help="Use user-level DP instead of record-level DP"
    )
    privacy_group.add_argument(
        "--no_dp", action="store_true",
        help="Disable DP sanitization (no clipping/noise) for sanity-check baselines"
    )

    # === Evaluation & Logging ===
    eval_group = p.add_argument_group('Evaluation & Logging')
    eval_group.add_argument(
        "--eval_every", type=int, default=5,
        help="Evaluate global model every N rounds"
    )
    eval_group.add_argument(
        "--save_results", action="store_true",
        help="Save training results to JSON file"
    )
    eval_group.add_argument(
        "--results_dir", type=str, default="./results",
        help="Directory for saving results"
    )
    eval_group.add_argument(
        "--verbose", action="store_true",
        help="Enable detailed logging and statistics"
    )

    # === System Configuration ===
    sys_group = p.add_argument_group('System')
    sys_group.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )

    return p.parse_args()


def validate_args(args) -> None:
    """Validate command line arguments for correctness."""

    # Privacy parameters
    # If --no_dp is set, we run a clean (non-private) baseline and do not validate ε/δ.
    if not getattr(args, "no_dp", False):
        if args.epsilon <= 0:
            raise ValueError("ε must be positive")
        if args.delta <= 0 or args.delta >= 1:
            raise ValueError("δ must be in (0, 1)")

    # Training parameters
    if args.client_lr <= 0 or args.server_lr <= 0:
        raise ValueError("Learning rates must be positive")
    if args.feature_clip_norm <= 0 or args.gradient_clip_norm <= 0:
        raise ValueError("Clipping norms must be positive")
    if args.batch_size <= 0:
        raise ValueError("Batch size must be positive")

    # SOFIM parameters
    if args.sofim_rho <= 0:
        raise ValueError("ρ must be positive")
    if args.sofim_beta < 0 or args.sofim_beta >= 1:
        raise ValueError("β must be in [0, 1)")

    # Federated learning parameters
    if args.num_clients <= 0:
        raise ValueError("Number of clients must be positive")
    if args.clients_per_round <= 0:
        raise ValueError("Clients per round must be positive")
    if args.clients_per_round > args.num_clients:
        raise ValueError("Clients per round cannot exceed total clients")
    if args.federated_rounds <= 0 or args.local_iterations <= 0:
        raise ValueError("Rounds and local iterations must be positive")

    # FedFC (Algorithm 3) validation
    if getattr(args, "use_fedfc", False):
        if args.fc_cc <= 0:
            raise ValueError("FedFC fc_cc must be positive")
        if args.fc_gamma <= 0:
            raise ValueError("FedFC fc_gamma must be positive")
        # If DP is enabled, sigma_c must be provided for the covariance release.
        if not getattr(args, "no_dp", False) and args.fc_sigma_c is None:
            raise ValueError("--use_fedfc requires --fc_sigma_c when DP is enabled")
        # Algorithm 3 uses all users per round; enforce consistency.
        if args.clients_per_round != args.num_clients:
            raise ValueError("For --use_fedfc (Algorithm 3), set --clients_per_round equal to --num_clients")
        # FedFC is mutually exclusive with FedNew-FC (different algorithms)
        if getattr(args, "use_fednew_fc", False):
            raise ValueError("Cannot use --use_fedfc and --use_fednew_fc together")

    # DP-SCAFFOLD validation
    if getattr(args, "use_scaffold", False):
        if args.scaffold_local_steps <= 0:
            raise ValueError("--scaffold_local_steps must be positive")
        if args.scaffold_client_lr <= 0:
            raise ValueError("--scaffold_client_lr must be positive")

        # Mutually exclusive with other algorithms
        if getattr(args, "use_sofim", False) or getattr(args, "use_fedfc", False) or getattr(args, "use_fednew_fc", False):
            raise ValueError("--use_scaffold cannot be combined with --use_sofim/--use_fedfc/--use_fednew_fc")

    # Data partitioning
    if args.partition_type == "non_iid_classes":
        # Max classes depends on dataset
        max_classes_by_dataset = {
            "cifar10": 10,
            "cifar10_binary": 2,
            "chestmnist": 14,  # MedMNIST default (multi-label); treated as upper bound for this check
            "pathmnist": 9,
            "bloodmnist": 8,
            "dermamnist": 7,
        }
        max_classes = max_classes_by_dataset.get(args.dataset, 10)

        if args.classes_per_client <= 0 or args.classes_per_client > max_classes:
            raise ValueError(
                f"Classes per client must be in [1, {max_classes}] for dataset={args.dataset}"
            )

    elif args.partition_type == "dirichlet":
        if args.dirichlet_alpha <= 0:
            raise ValueError("Dirichlet α must be positive")

    # ResNet-50 preprocessing options
    if args.resize_to_224 and args.no_resize_to_224:
        raise ValueError("Cannot specify both --resize_to_224 and --no_resize_to_224")


def setup_reproducibility(seed: int):
    """Setup random seeds for reproducible experiments."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic CUDA operations (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set Python hash seed for complete reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)


def print_experiment_header():
    """Print experiment header with algorithm information."""
    print("=" * 80)
    print("GRADIENT-BASED FEDERATED DP TRAINING")
    print("Second-Order Federated Learning (DP optional)")
    print("=" * 80)
    print("Algorithm: Client-side gradients + optional server optimizer (SOFIM / FedFC)")
    print("Privacy: DP can be enabled (Gaussian noise) or disabled (--no_dp)")
    print("Features: Frozen ResNet-18/50 → Linear classifier training only")
    print("=" * 80)


def print_configuration(args) -> None:
    """Print detailed experimental configuration."""
    print_experiment_header()

    # Task and model configuration
    task_type = "Binary" if args.binary else "10-way multiclass"
    print(f"\nTask: {task_type} classification on {args.dataset}")

    # Determine preprocessing pipeline
    if args.backbone == "resnet50":
        if args.no_resize_to_224:
            pipeline = "32×32 CIFAR-style stem"
            feature_dim = "2048-D"
        else:
            pipeline = "224×224 ImageNet-style stem (default)"
            feature_dim = "2048-D"
    else:
        pipeline = "32×32 CIFAR-style stem"
        feature_dim = "512-D"

    print(f"Backbone: {args.backbone.upper()} → {feature_dim} features")
    print(f"Preprocessing: {pipeline}")

    # Federated learning setup
    participation = 100 * args.clients_per_round / args.num_clients
    print(f"\nFederated Learning:")
    print(f"  • Total clients: {args.num_clients}")
    print(f"  • Clients per round: {args.clients_per_round} ({participation:.1f}%)")
    print(f"  • Communication rounds: {args.federated_rounds}")
    if getattr(args, "use_scaffold", False):
        print(f"  • Local steps (SCAFFOLD): {args.scaffold_local_steps}")
    else:
        print(f"  • Local iterations: {args.local_iterations} (gradient computations per round)")
    print(f"  • Client sampling: {args.client_sampling}")

    # Data distribution
    print(f"\nData Distribution:")
    print(f"  • Partition: {args.partition_type}")
    if args.partition_type == "non_iid_classes":
        print(f"  • Classes per client: {args.classes_per_client}")
    elif args.partition_type == "dirichlet":
        hetero_level = "High" if args.dirichlet_alpha < 0.5 else "Medium" if args.dirichlet_alpha < 1.0 else "Low"
        print(f"  • Dirichlet α: {args.dirichlet_alpha} ({hetero_level} heterogeneity)")

    # Training configuration
    print(f"\nTraining Configuration:")
    print(f"  • Batch size: {args.batch_size}")
    print(f"  • Client learning rate: {args.client_lr}")
    print(f"  • Server learning rate: {args.server_lr}")
    print(f"  • Feature clipping: L2 norm ≤ {args.feature_clip_norm} (Algorithm 2 Step 1)")
    print(f"  • Gradient clipping: L2 norm ≤ {args.gradient_clip_norm}")

    # Gradient-Based DP-SOFIM
    print(f"\nGradient-Based DP-SOFIM:")
    print(f"  • Client algorithm: Feature clipping → Gradient computation → DP noise")
    print(f"  • Server algorithm: Sherman-Morrison Newton on aggregated gradients")
    print(f"  • FIM regularization ρ: {args.sofim_rho}")
    print(f"  • Momentum parameter β: {args.sofim_beta}")
    print(f"  • Noise application: Single σ to gradients (Algorithm 2 Step 7)")

    # DP-FedFC (Algorithm 3)
    if getattr(args, "use_fedfc", False):
        print(f"\nDP-FedFC (Algorithm 3):")
        print(f"  • Enabled: True")
        print(f"  • Feature clip (C_c): {args.fc_cc}")
        print(f"  • Covariance noise (sigma_c): {args.fc_sigma_c if not getattr(args, 'no_dp', False) else 'N/A (no-dp)'}")
        print(f"  • Covariance regularization (gamma): {args.fc_gamma}")
        print(f"  • Participation: all clients per round (clients_per_round = num_clients)")

    # Privacy configuration
    dp_level = "User-level" if args.user_level_dp else "Record-level"
    print(f"\nDifferential Privacy:")
    if getattr(args, "no_dp", False):
        print("  • DP is DISABLED (--no_dp): no clipping, no noise, no accounting")
    else:
        print(f"  • Privacy budget: (ε={args.epsilon}, δ={args.delta})")
        print(f"  • Privacy model: {dp_level}")
        print(f"  • Mechanism: Gaussian noise calibrated by RDP accounting")

    # System information
    device = "GPU: " + torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU"
    print(f"\nSystem:")
    print(f"  • Device: {device}")
    print(f"  • Random seed: {args.seed}")
    print(f"  • PyTorch: {torch.__version__}")

    print("=" * 80)


def calculate_metrics(results: Dict[str, Any]) -> Dict[str, float]:
    """Calculate performance and efficiency metrics."""
    metrics = {}

    final_stats = results.get('final_stats', {})
    privacy_stats = results.get('privacy_stats', {})

    accuracy = final_stats.get('accuracy', 0)

    # If DP is disabled, do not compute privacy-derived metrics.
    # Prefer an explicit flag from training; fall back to absence of achieved_epsilon.
    dp_enabled = privacy_stats.get('dp_enabled', None)
    if dp_enabled is False:
        return metrics

    achieved_eps = privacy_stats.get('achieved_epsilon', 0)
    target_eps = privacy_stats.get('target_epsilon', 0)

    # Core metrics
    if accuracy > 0 and achieved_eps > 0:
        metrics['privacy_efficiency'] = accuracy / achieved_eps

    rounds = len(results.get('round_results', []))
    if rounds > 0 and accuracy > 0:
        metrics['communication_efficiency'] = accuracy / rounds

    if target_eps > 0 and achieved_eps > 0:
        metrics['privacy_utilization'] = achieved_eps / target_eps

    # Gradient-based specific metrics
    round_results = results.get('round_results', [])
    gradient_norms = [r.get('avg_gradient_norm', 0) for r in round_results if 'avg_gradient_norm' in r]
    if gradient_norms:
        metrics['avg_gradient_norm'] = np.mean(gradient_norms)
        metrics['gradient_std'] = np.std(gradient_norms)
        metrics['gradient_consistency'] = 1.0 / (1.0 + np.std(gradient_norms))

    return metrics


def save_experiment_results(args, results: Dict[str, Any], metrics: Dict[str, float]) -> None:
    """Save comprehensive experiment results."""
    if not args.save_results:
        return

    os.makedirs(args.results_dir, exist_ok=True)

    # Determine preprocessing configuration
    resize_to_224 = None
    if args.backbone == "resnet50":
        resize_to_224 = not args.no_resize_to_224

    # Create comprehensive results
    experiment_data = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),  # Timestamp stays in metadata for reference
            'experiment_type': 'gradient_based_federated_dp_sofim',
            'algorithm_version': 'v1.0',
            'dataset': args.dataset,
            'backbone': args.backbone,
            'preprocessing': {
                'resize_to_224': resize_to_224,
                'input_size': '224x224' if (args.backbone == 'resnet50' and resize_to_224) else '32x32'
            },
            'method_description': 'Client-side gradients with server-side SOFIM Newton'
        },
        'configuration': {
            'federated': {
                'num_clients': args.num_clients,
                'clients_per_round': args.clients_per_round,
                'federated_rounds': args.federated_rounds,
                'local_iterations': args.local_iterations,
                'client_sampling': args.client_sampling,
                'partition_type': args.partition_type,
                'classes_per_client': args.classes_per_client if args.partition_type == 'non_iid_classes' else None,
                'dirichlet_alpha': args.dirichlet_alpha if args.partition_type == 'dirichlet' else None
            },
            'gradient_based_sofim': {
                'rho': args.sofim_rho,
                'beta': args.sofim_beta,
                'feature_clip_norm': args.feature_clip_norm,
                'gradient_clip_norm': args.gradient_clip_norm,
                'client_side_method': 'gradient_computation_on_clipped_features',
                'server_side_method': 'sherman_morrison_newton'
            },
            'privacy': {
                'no_dp': bool(getattr(args, 'no_dp', False)),
                'epsilon': None if getattr(args, 'no_dp', False) else args.epsilon,
                'delta': None if getattr(args, 'no_dp', False) else args.delta,
                'user_level_dp': args.user_level_dp,
                'noise_mechanism': None if getattr(args, 'no_dp', False) else 'gaussian_on_gradients'
            },
            'training': {
                'client_lr': args.client_lr,
                'server_lr': args.server_lr,
                'batch_size': args.batch_size,
                'binary_classification': args.binary
            },
            'system': {
                'seed': args.seed,
                'device': torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU',
                'pytorch_version': torch.__version__
            }
        },
        'results': results,
        'derived_metrics': metrics
    }

    # Generate descriptive filename WITHOUT timestamp
    task = "binary" if args.binary else "multiclass"
    privacy_level = "user" if args.user_level_dp else "record"

    # Updated Filename: Includes 'rho' but removes 'timestamp'
    eps_tag = "nodp" if getattr(args, 'no_dp', False) else f"eps{args.epsilon}"
    filename = (f"gradient_fed_sofim_{args.backbone}_{task}_{args.partition_type}_"
                f"c{args.num_clients}_r{args.federated_rounds}_i{args.local_iterations}_"
                f"{eps_tag}_rho{args.sofim_rho}_{privacy_level}.json")

    filepath = os.path.join(args.results_dir, filename)

    # Overwrite mode ('w') ensures the old file with the same config is replaced
    with open(filepath, 'w') as f:
        json.dump(experiment_data, f, indent=2, default=str)

    print(f"\nExperiment results saved (overwritten): {filepath}")


def print_final_results(args, results: Dict[str, Any], metrics: Dict[str, float]) -> None:
    """Print comprehensive final results summary."""
    print("\n" + "=" * 80)
    print("TRAINING COMPLETED - GRADIENT-BASED DP-SOFIM RESULTS")
    print("=" * 80)

    # Extract key results
    final_stats = results.get('final_stats', {})
    privacy_stats = results.get('privacy_stats', {})

    accuracy = final_stats.get('accuracy', 0)
    loss = final_stats.get('loss', 0)
    dp_disabled = bool(getattr(args, 'no_dp', False))
    achieved_eps = privacy_stats.get('achieved_epsilon', None)
    target_eps = privacy_stats.get('target_epsilon', None if dp_disabled else args.epsilon)

    # Main performance metrics
    print(f"Final Performance:")
    print(f"  • Test Accuracy: {accuracy:.4f} ({100 * accuracy:.2f}%)")
    print(f"  • Test Loss: {loss:.4f}")
    if dp_disabled:
        print(f"  • Privacy: DISABLED (--no_dp)")
    else:
        # Fall back to args.epsilon if target_eps not present
        if target_eps is None:
            target_eps = args.epsilon
        # If achieved_eps is missing, print a placeholder
        if achieved_eps is None:
            print(f"  • Privacy Cost: ε = N/A (target: {target_eps})")
        else:
            print(f"  • Privacy Cost: ε = {achieved_eps:.4f} (target: {target_eps})")
        print(f"  • Privacy Level: {'User-level' if args.user_level_dp else 'Record-level'} DP")

    # Gradient-based algorithm analysis
    print(f"\nGradient-Based Algorithm Analysis:")
    if 'avg_gradient_norm' in metrics:
        print(f"  • Average gradient norm: {metrics['avg_gradient_norm']:.4f}")
    if 'gradient_std' in metrics:
        print(f"  • Gradient std deviation: {metrics['gradient_std']:.4f}")
    if 'gradient_consistency' in metrics:
        print(f"  • Gradient consistency: {metrics['gradient_consistency']:.4f}")

    # Efficiency metrics
    print(f"\nEfficiency Analysis:")
    if 'privacy_efficiency' in metrics:
        print(f"  • Privacy efficiency: {metrics['privacy_efficiency']:.4f} (accuracy/ε)")
    if 'communication_efficiency' in metrics:
        print(f"  • Communication efficiency: {metrics['communication_efficiency']:.4f} (accuracy/round)")
    if 'privacy_utilization' in metrics:
        print(f"  • Privacy budget usage: {100 * metrics['privacy_utilization']:.1f}%")

    # Training statistics
    total_time = final_stats.get('total_training_time', 0)
    avg_round_time = final_stats.get('avg_round_time', 0)

    print(f"\nTraining Statistics:")
    print(f"  • Total time: {total_time:.1f}s ({total_time / 60:.1f} min)")
    print(f"  • Average round time: {avg_round_time:.1f}s")
    print(f"  • Backbone: {args.backbone.upper()}")
    print(f"  • Local iterations per round: {args.local_iterations}")

    # SOFIM analysis
    research_metrics = results.get('research_metrics', {})
    if research_metrics:
        print(f"\nServer-Side SOFIM Newton Analysis:")
        if 'avg_momentum_norm' in research_metrics:
            print(f"  • Average momentum norm: {research_metrics['avg_momentum_norm']:.4f}")
        if 'avg_update_norm' in research_metrics:
            print(f"  • Average update norm: {research_metrics['avg_update_norm']:.4f}")

    # Federated learning summary
    participation_rate = 100 * args.clients_per_round / args.num_clients
    print(f"\nFederated Learning Summary:")
    print(f"  • Client participation rate: {participation_rate:.1f}%")
    print(f"  • Data distribution: {args.partition_type}")
    print(f"  • SOFIM parameters: ρ={args.sofim_rho}, β={args.sofim_beta}")
    print(f"  • Method: Gradient-Based with Server-side SOFIM Newton")

    print("=" * 80)


def main():
    """Main entry point for Gradient-Based Federated DP-SOFIM training."""
    start_time = time.time()

    try:
        # Parse and validate arguments
        args = parse_args()
        validate_args(args)
        # train.py expects args.learning_rate as the server/global LR
        if not hasattr(args, "learning_rate"):
            args.learning_rate = args.server_lr

        # Apply ResNet-50 default preprocessing
        if args.backbone == "resnet50" and not args.resize_to_224 and not args.no_resize_to_224:
            args.resize_to_224 = True

        # Setup reproducible environment
        setup_reproducibility(args.seed)

        # Print configuration
        if args.verbose:
            print_configuration(args)
        else:
            print_experiment_header()
            backbone_info = args.backbone.upper()
            if args.backbone == "resnet50":
                preprocessing = "224×224" if args.resize_to_224 else "32×32"
                backbone_info += f" ({preprocessing})"
            algo_tag = (
                "FedFC" if getattr(args, 'use_fedfc', False) else
                "FedNew-FC" if getattr(args, 'use_fednew_fc', False) else
                "SCAFFOLD" if getattr(args, 'use_scaffold', False) else
                "SOFIM" if getattr(args, 'use_sofim', False) else
                "FedGD"
            )
            if getattr(args, 'no_dp', False):
                print(f"Training: {args.num_clients} clients, {algo_tag}, no-DP baseline, {backbone_info}")
            else:
                print(f"Training: {args.num_clients} clients, {algo_tag}, ε={args.epsilon}, {backbone_info}")

        # Execute federated training
        print(f"\nInitializing Gradient-Based Federated DP-SOFIM training...")
        results = run_dpfedgd_training(args)

        if results is None:
            print("ERROR: Training failed to complete successfully")
            return 1

        # Calculate derived metrics
        derived_metrics = calculate_metrics(results)

        # Print comprehensive results
        print_final_results(args, results, derived_metrics)

        # Save experiment data
        save_experiment_results(args, results, derived_metrics)

        # Success summary
        final_accuracy = results.get('final_stats', {}).get('accuracy', 0)
        achieved_epsilon = results.get('privacy_stats', {}).get('achieved_epsilon', None)
        total_duration = time.time() - start_time

        if getattr(args, 'no_dp', False):
            print(f"\nSUCCESS: Accuracy={final_accuracy:.4f}, Privacy=DISABLED, Time={total_duration:.1f}s")
        else:
            # If achieved epsilon is missing, print a placeholder
            if achieved_epsilon is None:
                print(f"\nSUCCESS: Accuracy={final_accuracy:.4f}, ε=N/A, Time={total_duration:.1f}s")
            else:
                print(f"\nSUCCESS: Accuracy={final_accuracy:.4f}, ε={achieved_epsilon:.4f}, Time={total_duration:.1f}s")

        return 0

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        return 1

    except Exception as e:
        print(f"\nERROR: {e}")
        if hasattr(args, 'verbose') and args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
