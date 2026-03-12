"""
Record-Level DP-FedGD Sanitizer (Algorithm 5)

This implements the exact algorithm from the paper:
- Client-side: Clip per-example gradients, average, add Gaussian noise
- Server-side: Simple averaging and gradient descent update
# Privacy: Distributed Gaussian mechanism with exact hockey-stick DP accounting"""

import math
import torch
from torch import Tensor
from typing import List, Dict, Tuple, Optional
import numpy as np
from hockey_stick_accountant import RecordLevelDPFedGDAccountant, HockeyStickAccountant


def clip(gradient: Tensor, clip_norm: float = 1.0) -> Tensor:
    """L2 clipping: ensures gradient has norm <= clip_norm."""
    if gradient.dim() == 1:
        norm = gradient.norm(p=2)
        if norm > clip_norm:
            return gradient * (clip_norm / norm)
        return gradient
    else:
        # Batch of gradients - per-example clipping
        norms = gradient.norm(p=2, dim=1, keepdim=True)
        factors = (clip_norm / (norms + 1e-12)).clamp(max=1.0)
        return gradient * factors

class DPFedGDSanitizer:
    """
    Record-Level DP-FedGD Sanitizer implementing Algorithm 5.

    Algorithm 5 breakdown:
    - Input: n clients, noise level σ_g, clipping constant C_g, training length T, learning rate η
    - For each round k:
      * Each client i clips and averages gradients, then adds noise:
        u_i^k = (Σ clip_C(∇f(θ^k, x))) / |D_i| + E_i^k
        where E_i^k ~ N(0, I_d(C_g σ_g)²/n)
      * Server aggregates: U^k = (Σ u_i^k) / n
      * Update: θ^(k+1) = θ^k - η · U^k
    """

    def __init__(self,
                 num_clients: int,
                 clients_per_round: int,
                 federated_rounds: int,
                 dataset_size: int,
                 gradient_clip_norm: float,
                 sigma_g: float = None,
                 neighbor_type: str = "replace_one",
                 display_rounds: Optional[int] = None,
                 device: torch.device = torch.device('cpu')):
        """
        Initialize DP-FedGD sanitizer.

        Args:
            num_clients: Total number of clients (n in algorithm)
            clients_per_round: Number of clients participating per round
            federated_rounds: Total number of training rounds (T in algorithm)
            gradient_clip_norm: Clipping constant C_g
            sigma_g: Noise multiplier σ_g (if None, will be calibrated)
            device: Computing device
        """
        self.num_clients = num_clients
        self.clients_per_round = clients_per_round
        self.federated_rounds = federated_rounds
        # If provided, this is the number of *training* rounds shown in logs.
        # `self.federated_rounds` may be larger when we calibrate privacy for multiple DP releases per round.
        self.display_rounds = display_rounds if display_rounds is not None else federated_rounds
        self.dataset_size = dataset_size
        self.gradient_clip_norm = gradient_clip_norm  # C_g
        self.sigma_g = sigma_g  # σ_g
        self.device = device

        # Privacy tracking
        self.epsilon = None
        self.delta = None
        self.hockey_accountant = RecordLevelDPFedGDAccountant(
            C_g=gradient_clip_norm,
            n_clients=clients_per_round,
            T_rounds=federated_rounds,
            dataset_size=dataset_size,
            neighbor_type=neighbor_type
        )
        self.current_round = 0

        # Statistics
        self.gradient_norms = []
        self.noise_norms = []

    def get_sum_noise_std(self) -> float:
        """Std of noise added to the SUM before division."""
        return (self.gradient_clip_norm * self.sigma_g) / np.sqrt(self.clients_per_round)

    def get_update_noise_std(self, client_dataset_size: int = None) -> float:
        """Std of noise in the released averaged update."""
        if client_dataset_size is None:
            client_dataset_size = self.dataset_size
        return self.get_sum_noise_std() / client_dataset_size

    def clip_gradients_batch(self, gradients: Tensor) -> Tensor:
        """
        Clip per-example gradients to have norm <= C_g.

        Args:
            gradients: Tensor of shape [batch_size, d] containing per-example gradients

        Returns:
            Clipped gradients of same shape
        """
        return clip(gradients, self.gradient_clip_norm)

    def add_client_noise(self, clipped_sum: Tensor, client_dataset_size: int = None) -> Tensor:
        if client_dataset_size is None:
            client_dataset_size = self.dataset_size  # fallback

        # Noise on SUM:  (C_g σ_g / √n)
        sum_noise_std = (self.gradient_clip_norm * self.sigma_g) / math.sqrt(self.clients_per_round)
        noise = torch.randn_like(clipped_sum) * sum_noise_std
        update_noise_std = sum_noise_std / client_dataset_size
        server_noise_std = sum_noise_std / (client_dataset_size * self.clients_per_round)

        noisy_update = (clipped_sum + noise) / client_dataset_size
        self.gradient_norms.append(clipped_sum.norm().item())
        self.noise_norms.append(noise.norm().item())
        cos_sim = torch.nn.functional.cosine_similarity(
            clipped_sum.flatten(),
            (clipped_sum + noise).flatten(),
            dim=0
        ).item()

        stats = {
            "sum_noise_std": float(sum_noise_std),
            "update_noise_std": float(update_noise_std),
            "server_noise_std": float(server_noise_std),
            "noise_norm": float(noise.norm().item()),
            "clipped_sum_norm": float(clipped_sum.norm().item()),
            "noisy_update_norm": float(noisy_update.norm().item()),
            "client_dataset_size": int(client_dataset_size),
            "cosine_signal_noise": float(cos_sim),
        }
        return noisy_update, stats

    def aggregate_client_updates(self, client_updates: List[Tensor]) -> Tensor:
        """
        Server aggregation step from Algorithm 5.

        Server aggregates {u_i^k}_{i=1}^n and computes: U^k = (Σ u_i^k) / n

        Args:
            client_updates: List of noisy client updates

        Returns:
            Aggregated update U^k
        """
        if not client_updates:
            raise ValueError("No client updates to aggregate")

        # Simple averaging: U^k = (Σ u_i^k) / n
        aggregated = torch.stack(client_updates).mean(dim=0)
        return aggregated

    def calibrate_noise(self, target_epsilon: float, target_delta: float):
        """
        Calibrate noise multiplier σ_g using hockey-stick accountant.
        """
        self.epsilon = target_epsilon
        self.delta = target_delta

        if target_epsilon >= 500.0:
            self.sigma_g = 1e-10
            print(f"Large epsilon ({target_epsilon}), using minimal noise")
            return

        # Use hockey-stick accountant to calibrate
        self.sigma_g = self.hockey_accountant.calibrate_noise(
            epsilon_target=target_epsilon,
            delta_target=target_delta
        )

    def start_round(self):
        """Mark the start of a new training round."""
        self.current_round += 1
        if self.current_round % 10 == 0:
            print(f"  Round {self.current_round}/{self.display_rounds}")

    def get_privacy_cost(self):
        if self.sigma_g is None or self.delta is None:
            return None, None
        achieved_eps = self.hockey_accountant.get_privacy_cost(delta=self.delta)
        return float(achieved_eps), float(self.delta)

    def get_stats(self) -> Dict:
        """Get statistics about the training process."""
        return {
            'configuration': {
                'num_clients': self.num_clients,
                'clients_per_round': self.clients_per_round,
                'federated_rounds': self.federated_rounds,
                'gradient_clip_norm': self.gradient_clip_norm,
                'sigma_g': self.sigma_g,
                'client_sum_noise_std': self.get_sum_noise_std()
            },
            'privacy': {
                'epsilon': self.epsilon,
                'delta': self.delta,
                'current_round': self.current_round
            },
            'statistics': {
                'gradient_norms': self.gradient_norms.copy(),
                'noise_norms': self.noise_norms.copy()
            }
        }

class DPFedScaffoldSanitizer(DPFedGDSanitizer):
    """
    Sanitizer for record-level DP-SCAFFOLD local-step releases.

    DP-SCAFFOLD releases a noisy averaged gradient at EACH local step.
    Under replace-one adjacency, the L2 sensitivity of a clipped SUM is <= 2*C_g.
    """

    def __init__(
        self,
        num_clients: int,
        clients_per_round: int,
        federated_rounds: int,
        dataset_size: int,
        gradient_clip_norm: float,
        sigma_g: float = None,
        neighbor_type: str = "replace_one",
        display_rounds: Optional[int] = None,
        device: torch.device = torch.device("cpu"),
        sensitivity_factor: Optional[float] = None,
    ):
        super().__init__(
            num_clients=num_clients,
            clients_per_round=clients_per_round,
            federated_rounds=federated_rounds,
            dataset_size=dataset_size,
            gradient_clip_norm=gradient_clip_norm,
            sigma_g=sigma_g,
            neighbor_type=neighbor_type,
            display_rounds=display_rounds,
            device=device,
        )

        if sensitivity_factor is None:
            self.sensitivity_factor = 2.0 if neighbor_type == "replace_one" else 1.0
        else:
            self.sensitivity_factor = float(sensitivity_factor)

    def get_sum_noise_std(self) -> float:
        return (self.sensitivity_factor * self.gradient_clip_norm * self.sigma_g) / np.sqrt(self.clients_per_round)

    def get_local_step_update_noise_std(self, client_dataset_size: int) -> float:
        return self.get_sum_noise_std() / max(1, client_dataset_size)

    def add_client_noise(self, clipped_sum: Tensor, client_dataset_size: int = None) -> Tensor:
        if client_dataset_size is None:
            client_dataset_size = self.dataset_size

        sum_noise_std = (self.sensitivity_factor * self.gradient_clip_norm * self.sigma_g) / math.sqrt(self.clients_per_round)
        noise = torch.randn_like(clipped_sum) * sum_noise_std

        update_noise_std = sum_noise_std / client_dataset_size
        server_noise_std = sum_noise_std / (client_dataset_size * self.clients_per_round)

        noisy_update = (clipped_sum + noise) / client_dataset_size

        self.gradient_norms.append(clipped_sum.norm().item())
        self.noise_norms.append(noise.norm().item())

        cos_sim = torch.nn.functional.cosine_similarity(
            clipped_sum.flatten(), (clipped_sum + noise).flatten(), dim=0
        ).item()

        stats = {
            "sum_noise_std": float(sum_noise_std),
            "update_noise_std": float(update_noise_std),
            "server_noise_std": float(server_noise_std),
            "noise_norm": float(noise.norm().item()),
            "clipped_sum_norm": float(clipped_sum.norm().item()),
            "noisy_update_norm": float(noisy_update.norm().item()),
            "client_dataset_size": int(client_dataset_size),
            "cosine_signal_noise": float(cos_sim),
        }
        return noisy_update, stats

class DPFedNewSanitizer:
    """
    Sanitizer for DP-FedNew-FC.
    Directly calibrates noise for the primal update sensitivity Delta_H = C2 / gamma.
    """

    def __init__(self, num_clients: int, clients_per_round: int,
                 federated_rounds: int, fn_c2: float, fn_gamma: float,
                 fn_c_primal: float, device: torch.device):
        self.num_clients = num_clients
        self.clients_per_round = clients_per_round
        self.federated_rounds = federated_rounds
        self.fn_c2 = fn_c2
        self.fn_gamma = fn_gamma
        self.fn_c_primal = fn_c_primal
        self.device = device

        #
        # The record-level sensitivity of the primal update is Delta_H = C2 / gamma
        self.sensitivity = fn_c2 / fn_gamma

        # Use the base HockeyStickAccountant directly to avoid gradient-specific assumptions
        # (like the 2*C/n factor in RecordLevelDPFedGDAccountant)
        self.accountant = HockeyStickAccountant()

        self.sigma_h = 0.0  # This is the noise multiplier calibrated for Delta_H
        self.current_round = 0
        self.epsilon = None
        self.delta = None

    def calibrate_noise(self, target_epsilon: float, target_delta: float):
        """Finds noise multiplier sigma_h such that T-round composition yields (eps, delta)."""
        self.epsilon = target_epsilon
        self.delta = target_delta

        # Calibrate for a mechanism that releases the aggregate with sensitivity self.sensitivity
        #
        self.sigma_h = self.accountant.calibrate_noise(
            epsilon_target=target_epsilon,
            delta_target=target_delta,
            Delta=self.sensitivity,
            T=self.federated_rounds
        )

    def sanitize_primal_update(self, y_hat: Tensor) -> Tensor:
        """
        Algorithm 1 Step 13: Add DP noise to the preconditioned primal variable.
        Includes 1/sqrt(n) scaling to ensure server-side average has correct noise.
        """
        # 1. Hard cap clipping (C) to ensure the release is bounded
        #
        y_bar = y_hat
        y_norm = y_bar.norm(p=2).item()
        if y_norm > self.fn_c_primal:
            y_bar = y_bar * (self.fn_c_primal / (y_norm + 1e-12))

        # 2. Scaling adjustment for Distributed DP:
        # To ensure the average at the server has std = sigma_h * sensitivity,
        # each client adds noise with std scaled by 1/sqrt(K).
        distributed_scale = 1.0 / math.sqrt(self.clients_per_round)
        noise_std = self.sigma_h * self.sensitivity * distributed_scale

        noise = torch.randn_like(y_bar) * noise_std
        return y_bar + noise

    def compute_lemma4_xi(self, g_clean: Tensor, b_admm: Tensor) -> float:
        """
        Computes scalar xi such that ||g_clean + xi*b_admm||_2 = C2.
        Picks the root that places the update on the boundary.
        """
        #
        # ||a + xi*b||^2 = C2^2 => xi^2||b||^2 + 2*xi<a,b> + ||a||^2 - C2^2 = 0
        a = g_clean
        b = b_admm

        b_norm_sq = (b * b).sum().item()
        if b_norm_sq < 1e-12:
            return 1.0

        a_norm_sq = (a * a).sum().item()
        dot_ab = (a * b).sum().item()

        A = b_norm_sq
        B = 2 * dot_ab
        C = a_norm_sq - self.fn_c2 ** 2

        discriminant = B ** 2 - 4 * A * C
        if discriminant < 0:
            # Numerically g_clean might already be outside or close to the boundary
            return self.fn_c2 / (g_clean.norm().item() + 1e-12)

        # Standard quadratic formula roots
        sqrt_disc = math.sqrt(discriminant)
        xi1 = (-B + sqrt_disc) / (2 * A)
        xi2 = (-B - sqrt_disc) / (2 * A)

        # We want the root that moves g_clean along b_admm toward the boundary
        # Picking the larger xi ensures we reach the C2 radius.
        return max(xi1, xi2)

    def scale_auxiliary_gradient(self, g_clean: Tensor, b_admm: Tensor) -> Tensor:
        """Step 8-11: Scale the auxiliary gradient to the C2 ball."""
        g_sum = g_clean + b_admm
        if g_sum.norm(p=2) > self.fn_c2:
            xi = self.compute_lemma4_xi(g_clean, b_admm)
            return g_clean + xi * b_admm
        return g_sum

    def get_client_noise_std(self) -> float:
        """Helper for reporting per-client noise level."""
        distributed_scale = 1.0 / math.sqrt(self.clients_per_round)
        return self.sigma_h * self.sensitivity * distributed_scale

    def start_round(self):
        self.current_round += 1

    def get_privacy_cost(self) -> Tuple[float, float]:
        return self.epsilon, self.delta


class DPFedFCSanitizer(DPFedGDSanitizer):
    """
    Sanitizer for Record-level DP-FedFC.
    Handles noise for both the covariance matrix (sigma_c) and gradients (sigma_g).
    """
    def __init__(self, *args, fn_cc: float, fn_cg: float, sigma_c: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.fn_cc = fn_cc  # Clipping constant for features Cc
        self.fn_cg = fn_cg  # Clipping constant for gradients Cg
        self.sigma_c = sigma_c # Noise level for covariance

        # Ensure the gradient clipping constant used by inherited methods matches Cg
        self.gradient_clip_norm = self.fn_cg

        # Algorithm 3 uses n = total number of users (all users participate).
        # Override the accountant from the base class so its n_clients matches num_clients.
        self.hockey_accountant = RecordLevelDPFedGDAccountant(
            C_g=self.gradient_clip_norm,
            n_clients=self.num_clients,
            T_rounds=self.federated_rounds,
            dataset_size=self.dataset_size,
            neighbor_type="replace_one"
        )

    def add_covariance_noise(self, local_covariance: torch.Tensor) -> torch.Tensor:
        """
        Algorithm 3 Line 6: Add noise with variance (Cc^2 * sigma_c)^2 / n
        Standard deviation is (Cc^2 * sigma_c) / sqrt(n)
        """
        # Use total participants (n) for the initial covariance release
        dist_scale = 1.0 / math.sqrt(self.num_clients)
        std = (self.fn_cc ** 2 * self.sigma_c) * dist_scale

        noise = torch.randn_like(local_covariance) * std
        return local_covariance + noise

    def add_client_noise_fedfc(self, clipped_sum: torch.Tensor, client_dataset_size: int) -> torch.Tensor:
        """Algorithm 3 (DP-FedFC) Line 11: gradient perturbation.

        The paper defines:
            u_i^k = (\sum_x clip_{Cg}(\nabla f(\theta^k, x)) + E_i^k) / |D_i|
            E_i^k ~ N(0, I_d (Cg * sigma_g)^2 / n)

        Therefore, noise is added to the *SUM* with std (Cg * sigma_g) / sqrt(n),
        and only then divided by |D_i|.

        Notes:
        - Algorithm 3 uses n = total number of users; we use self.num_clients.
        """
        if client_dataset_size is None:
            client_dataset_size = self.dataset_size

        # Noise on SUM: std = (Cg * sigma_g) / sqrt(n)
        sum_noise_std = (self.gradient_clip_norm * self.sigma_g) / math.sqrt(self.num_clients)
        noise = torch.randn_like(clipped_sum) * sum_noise_std

        # Divide AFTER noise by |D_i|
        return (clipped_sum + noise) / client_dataset_size

    def add_client_noise(self, clipped_sum: Tensor, client_dataset_size: int = None):
        """Override base DP-FedGD noise to match DP-FedFC Algorithm 3 scaling.

        Algorithm 3 uses n = total number of users (full participation).
        Noise is added to the SUM with std (Cg * sigma_g) / sqrt(n), then divided by |D_i|.

        Returns:
            (noisy_update, stats_dict)
        """
        if client_dataset_size is None:
            client_dataset_size = self.dataset_size

        # Noise on SUM: std = (Cg * sigma_g) / sqrt(n)
        sum_noise_std = (self.gradient_clip_norm * self.sigma_g) / math.sqrt(self.num_clients)
        noise = torch.randn_like(clipped_sum) * sum_noise_std
        noisy_update = (clipped_sum + noise) / client_dataset_size

        stats = {
            "sum_noise_std": float(sum_noise_std),
            "update_noise_std": float(sum_noise_std / client_dataset_size),
            "server_noise_std": float(sum_noise_std / (client_dataset_size * self.num_clients)),
            "noise_norm": float(noise.norm().item()),
            "clipped_sum_norm": float(clipped_sum.norm().item()),
            "noisy_update_norm": float(noisy_update.norm().item()),
            "client_dataset_size": int(client_dataset_size),
        }
        return noisy_update, stats
