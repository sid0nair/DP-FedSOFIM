"""
Exact Hockey-Stick Divergence (Gaussian DP) Accountant
Following the implementation guidelines from the paper

This implements the exact privacy accounting using the closed-form
hockey-stick divergence formula for composition of Gaussian mechanisms.

Reference: "Communication-Efficient Differentially Private Federated Learning
Using Second-Order Information" (ICLR 2024)
"""

import numpy as np
from scipy.stats import norm
from typing import Tuple, Optional
import warnings


class HockeyStickAccountant:
    """
    Exact Gaussian DP accountant using hockey-stick divergence formula.

    Implements Algorithm 2 and Algorithm 3 from the implementation guide:
    - Algorithm 2: Compute δ(ε) for T compositions
    - Algorithm 3: Find ε for target δ via binary search
    """

    def __init__(self):
        """Initialize the accountant."""
        self.Delta = None  # L2 sensitivity of released quantity
        self.sigma = None  # Noise std of released quantity
        self.T = None  # Number of compositions (rounds)

    @staticmethod
    def compute_delta(epsilon: float, Delta: float, sigma: float, T: int) -> float:
        """
        Algorithm 2: Exact Gaussian DP δ(ε) for T compositions.

        For composition of T Gaussian mechanisms with sensitivity Δ and noise std σ:

        δ(ε) = Φ(−εσ/(√T·Δ) + √T·Δ/(2σ)) − e^ε · Φ(−εσ/(√T·Δ) − √T·Δ/(2σ))

        Args:
            epsilon: Privacy parameter ε
            Delta: L2 sensitivity Δ of each release
            sigma: Noise standard deviation σ of each release
            T: Number of compositions (communication rounds)

        Returns:
            delta: Privacy parameter δ(ε)
        """
        if sigma <= 0:
            return 1.0  # No noise = no privacy

        if T <= 0:
            return 0.0  # No releases = perfect privacy

        # Numerically stable implementation using a and b
        sqrt_T = np.sqrt(T)

        # a = εσ/(√T·Δ)
        a = epsilon * sigma / (sqrt_T * Delta)

        # b = √T·Δ/(2σ)
        b = sqrt_T * Delta / (2.0 * sigma)

        # δ(ε) = Φ(−a + b) − e^ε · Φ(−a − b)
        term1 = norm.cdf(-a + b)
        term2 = np.exp(epsilon) * norm.cdf(-a - b)

        delta = term1 - term2

        # Clip to avoid tiny negative values from floating point errors
        return max(delta, 0.0)

    @staticmethod
    def find_epsilon(delta_target: float, Delta: float, sigma: float, T: int,
                     epsilon_max: float = 100.0, num_iters: int = 100,
                     tolerance: float = 1e-8) -> float:
        """
        Algorithm 3: Find ε for target δ via binary search.

        Finds the smallest ε such that δ(ε) ≤ δ_target.

        Args:
            delta_target: Target privacy parameter δ
            Delta: L2 sensitivity Δ of each release
            sigma: Noise standard deviation σ of each release
            T: Number of compositions (communication rounds)
            epsilon_max: Upper bound for binary search (default: 100.0)
            num_iters: Number of binary search iterations (default: 100)
            tolerance: Convergence tolerance (default: 1e-8)

        Returns:
            epsilon: Privacy parameter ε achieving δ(ε) ≤ δ_target
        """
        # Binary search bounds
        eps_lower = 0.0
        eps_upper = epsilon_max

        # Check if achievable
        delta_at_max = HockeyStickAccountant.compute_delta(
            epsilon_max, Delta, sigma, T
        )

        if delta_at_max > delta_target:
            warnings.warn(
                f"Cannot achieve δ={delta_target} with ε≤{epsilon_max}. "
                f"At ε={epsilon_max}, δ={delta_at_max:.2e}. "
                f"Returning ε={epsilon_max}."
            )
            return epsilon_max

        # Binary search
        for iteration in range(num_iters):
            eps_mid = (eps_lower + eps_upper) / 2.0

            delta_mid = HockeyStickAccountant.compute_delta(
                eps_mid, Delta, sigma, T
            )

            # Check convergence
            if abs(delta_mid - delta_target) < tolerance:
                return eps_mid

            # Update bounds
            # Note: δ(ε) is DECREASING in ε
            if delta_mid > delta_target:
                # Need smaller δ, so need larger ε
                eps_lower = eps_mid
            else:
                # δ is small enough, try smaller ε
                eps_upper = eps_mid

        # Return upper bound (conservative)
        return eps_upper

    def set_mechanism_parameters(self, Delta: float, sigma: float, T: int):
        """
        Set the mechanism parameters for privacy accounting.

        Args:
            Delta: L2 sensitivity of released quantity
            sigma: Noise standard deviation of released quantity
            T: Number of communication rounds (releases)
        """
        self.Delta = Delta
        self.sigma = sigma
        self.T = T

    def get_epsilon(self, delta: float) -> float:
        """
        Compute ε for given δ using current mechanism parameters.

        Args:
            delta: Target privacy parameter δ

        Returns:
            epsilon: Privacy parameter ε achieving (ε, δ)-DP
        """
        if self.Delta is None or self.sigma is None or self.T is None:
            raise ValueError("Must call set_mechanism_parameters first")

        return self.find_epsilon(delta, self.Delta, self.sigma, self.T)

    def calibrate_noise(self, epsilon_target: float, delta_target: float,
                        Delta: float, T: int,
                        sigma_min: float = 1e-6, sigma_max: float = 5000.0,
                        num_iters: int = 100, tolerance: float = 1e-12) -> float:
        """
        Generic calibration: Finds noise multiplier 'sigma' such that T rounds
        achieve (eps, delta)-DP for a release with sensitivity 'Delta'.
        """
        low = sigma_min
        high = sigma_max

        # Check if target is achievable
        if self.compute_delta(epsilon_target, Delta, high, T) > delta_target:
            return high

        for _ in range(num_iters):
            mid = (low + high) / 2.0
            # achieved_delta decreases as noise (mid) increases
            achieved_delta = self.compute_delta(epsilon_target, Delta, mid, T)

            if abs(achieved_delta - delta_target) < tolerance:
                return mid

            if achieved_delta > delta_target:
                low = mid
            else:
                high = mid

        return high

    def get_delta(self, epsilon: float) -> float:
        """
        Compute δ for given ε using current mechanism parameters.

        Args:
            epsilon: Privacy parameter ε

        Returns:
            delta: Privacy parameter δ for (ε, δ)-DP
        """
        if self.Delta is None or self.sigma is None or self.T is None:
            raise ValueError("Must call set_mechanism_parameters first")

        return self.compute_delta(epsilon, self.Delta, self.sigma, self.T)


class RecordLevelDPFedGDAccountant:
    """
    Privacy accountant specifically for Record-Level DP-FedGD (Algorithm 5).

    Handles the specific setup:
    - Per-example gradient clipping with bound C_g
    - Gaussian noise with std (C_g·σ_g)/√n per client
    - T communication rounds
    - All clients participate each round
    """

    def __init__(self, C_g: float, n_clients: int, T_rounds: int,
                 dataset_size: int, neighbor_type: str = "replace_one"):
        """
        Initialize accountant for Record-Level DP-FedGD.

        Args:
            C_g: Gradient clipping norm
            n_clients: Number of clients per round (n)
            T_rounds: Number of communication rounds (T)
            dataset_size: Size of each client's dataset |D_i|
            neighbor_type: "replace_one" or "add_remove_one"
        """
        self.C_g = C_g
        self.n_clients = n_clients
        self.T_rounds = T_rounds
        self.dataset_size = dataset_size
        self.neighbor_type = neighbor_type

        self.accountant = HockeyStickAccountant()
        self.sigma_g = None  # Will be calibrated

    def _compute_sensitivity_and_noise(self, sigma_g: float) -> Tuple[float, float]:
        """
        Compute sensitivity Δ and noise std σ for the released quantity u_i^k.

        Following Section 2.3 and 2.2 of the implementation guide.

        Returns:
            (Delta, sigma): Sensitivity and noise std of released u_i^k
        """
        # Released quantity: u_i^k = (S_i^k + E_i^k) / |D_i|
        # where S_i^k = sum of clipped per-example gradients
        #       E_i^k ~ N(0, (C_g·σ_g)²/n · I_d)

        # Noise std of released quantity (Section 2.2)
        # σ_release = C_g·σ_g / (√n · |D_i|)
        sigma_release = (self.C_g * sigma_g) / (np.sqrt(self.n_clients) * self.dataset_size)

        # Sensitivity of released quantity (Section 2.3)
        if self.neighbor_type == "replace_one":
            # Replace-one: changing one record changes clipped sum by ≤ 2·C_g
            # After dividing by |D_i|: Δ = 2·C_g / |D_i|
            Delta = (2.0 * self.C_g) / self.dataset_size
        elif self.neighbor_type == "add_remove_one":
            # Add/remove-one: typically C_g / |D_i|
            Delta = self.C_g / self.dataset_size
        else:
            raise ValueError(f"Unknown neighbor_type: {self.neighbor_type}")

        return Delta, sigma_release

    def calibrate_noise(self, epsilon_target: float, delta_target: float,
                        sigma_g_max: float = 5000.0, num_iters: int = 100,
                        tolerance: float = 1e-10) -> float:
        """
        Calibrate σ_g to achieve target (ε, δ)-DP.

        Uses binary search to find σ_g such that T rounds achieve
        (ε_target, δ_target)-DP.

        Args:
            epsilon_target: Target privacy parameter ε
            delta_target: Target privacy parameter δ
            sigma_g_max: Upper bound for σ_g search
            num_iters: Number of binary search iterations
            tolerance: Convergence tolerance

        Returns:
            sigma_g: Calibrated noise multiplier
        """
        print(f"\nCalibrating noise for Record-Level DP-FedGD:")
        print(f"  Target: ε={epsilon_target:.4f}, δ={delta_target}")
        print(f"  Clients per round: {self.n_clients}")
        print(f"  Training rounds: {self.T_rounds}")
        print(f"  Gradient clip norm: {self.C_g}")
        print(f"  Dataset size per client: {self.dataset_size}")
        print(f"  Neighbor type: {self.neighbor_type}")

        # Binary search for σ_g
        sigma_lower = 0.01
        sigma_upper = sigma_g_max

        for iteration in range(num_iters):
            sigma_mid = (sigma_lower + sigma_upper) / 2.0

            # Compute sensitivity and noise for this σ_g
            Delta, sigma_release = self._compute_sensitivity_and_noise(sigma_mid)

            # Compute achieved epsilon using hockey-stick formula
            achieved_delta = HockeyStickAccountant.compute_delta(
                epsilon_target, Delta, sigma_release, self.T_rounds
            )

            if abs(achieved_delta - delta_target) < tolerance:
                print(f"  Converged after {iteration + 1} iterations")
                break

            # Update search bounds
            # δ(ε) decreases as noise increases (σ increases)
            if achieved_delta > delta_target:
                # Need more noise (larger σ_g)
                sigma_lower = sigma_mid
            else:
                # Have enough noise (can decrease σ_g)
                sigma_upper = sigma_mid

        self.sigma_g = sigma_mid

        # Final verification
        Delta, sigma_release = self._compute_sensitivity_and_noise(self.sigma_g)
        final_delta = HockeyStickAccountant.compute_delta(
            epsilon_target, Delta, sigma_release, self.T_rounds
        )

        # Also compute what epsilon we achieve for the target delta
        final_epsilon = HockeyStickAccountant.find_epsilon(
            delta_target, Delta, sigma_release, self.T_rounds
        )

        client_noise_std = (self.C_g * self.sigma_g) / np.sqrt(self.n_clients)

        print(f"\nNoise Calibration Complete:")
        print(f"  Noise multiplier σ_g: {self.sigma_g:.6f}")
        print(f"  Per-client noise std: {client_noise_std:.6f}")
        print(f"  Sensitivity Δ: {Delta:.6e}")
        print(f"  Noise std σ (released): {sigma_release:.6e}")
        print(f"  Achieved δ at ε={epsilon_target}: {final_delta:.6e}")
        print(f"  Achieved ε at δ={delta_target}: {final_epsilon:.6f}")

        return self.sigma_g

    def get_client_noise_std(self) -> float:
        """
        Get the per-client noise standard deviation.

        Each client adds E_i^k ~ N(0, (C_g·σ_g/√n)² · I_d).

        Returns:
            std: Standard deviation per coordinate = (C_g·σ_g)/√n
        """
        if self.sigma_g is None:
            raise ValueError("Must call calibrate_noise first")

        return (self.C_g * self.sigma_g) / np.sqrt(self.n_clients)

    def get_privacy_cost(self, delta: float) -> float:
        """
        Compute achieved ε for given δ.

        Args:
            delta: Privacy parameter δ

        Returns:
            epsilon: Privacy parameter ε
        """
        if self.sigma_g is None:
            raise ValueError("Must call calibrate_noise first")

        Delta, sigma_release = self._compute_sensitivity_and_noise(self.sigma_g)

        return HockeyStickAccountant.find_epsilon(
            delta, Delta, sigma_release, self.T_rounds
        )

    def verify_privacy(self, epsilon: float, delta: float) -> dict:
        """
        Verify achieved privacy parameters.

        Args:
            epsilon: Privacy parameter ε to check
            delta: Privacy parameter δ to check

        Returns:
            dict with verification results
        """
        if self.sigma_g is None:
            raise ValueError("Must call calibrate_noise first")

        Delta, sigma_release = self._compute_sensitivity_and_noise(self.sigma_g)

        achieved_delta = HockeyStickAccountant.compute_delta(
            epsilon, Delta, sigma_release, self.T_rounds
        )

        achieved_epsilon = HockeyStickAccountant.find_epsilon(
            delta, Delta, sigma_release, self.T_rounds
        )

        return {
            'sigma_g': self.sigma_g,
            'client_noise_std': self.get_client_noise_std(),
            'sensitivity_Delta': Delta,
            'noise_std_sigma': sigma_release,
            'target_epsilon': epsilon,
            'achieved_delta_at_target_epsilon': achieved_delta,
            'satisfies_epsilon': achieved_delta <= delta,
            'target_delta': delta,
            'achieved_epsilon_at_target_delta': achieved_epsilon,
            'satisfies_delta': achieved_epsilon <= epsilon,
        }


# Sanity tests (as specified in Section 6 of the guide)
def run_sanity_tests():
    """Run sanity tests on the hockey-stick accountant."""
    print("=" * 70)
    print("Running Sanity Tests")
    print("=" * 70)

    # Test 1: T=1 reduces to single-mechanism expression
    print("\nTest 1: T=1 (single mechanism)")
    Delta, sigma, T = 1.0, 1.0, 1
    epsilon = 1.0
    delta = HockeyStickAccountant.compute_delta(epsilon, Delta, sigma, T)
    print(f"  ε={epsilon}, Δ={Delta}, σ={sigma}, T={T}")
    print(f"  δ(ε) = {delta:.6e}")

    # Test 2: σ → ∞ yields δ → 0
    print("\nTest 2: Large noise (σ → ∞)")
    Delta, T = 1.0, 10
    for sigma in [10, 100, 1000]:
        delta = HockeyStickAccountant.compute_delta(epsilon, Delta, sigma, T)
        print(f"  σ={sigma:4d}: δ(ε) = {delta:.6e}")

    # Test 3: δ(ε) decreases as ε increases
    print("\nTest 3: δ(ε) monotonicity (should decrease as ε increases)")
    Delta, sigma, T = 1.0, 1.0, 10
    prev_delta = float('inf')
    for epsilon in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
        delta = HockeyStickAccountant.compute_delta(epsilon, Delta, sigma, T)
        is_decreasing = delta < prev_delta
        print(f"  ε={epsilon:4.1f}: δ={delta:.6e} (decreasing: {is_decreasing})")
        prev_delta = delta

    print("\n" + "=" * 70)
    print("Sanity Tests Complete")
    print("=" * 70)


if __name__ == "__main__":
    # Run sanity tests
    run_sanity_tests()

    # Example: Calibrate for Record-Level DP-FedGD
    print("\n\nExample: Record-Level DP-FedGD")
    print("=" * 70)

    accountant = RecordLevelDPFedGDAccountant(
        C_g=1.0,  # Clipping norm
        n_clients=20,  # Clients per round
        T_rounds=100,  # Training rounds
        dataset_size=100,  # Samples per client
        neighbor_type="replace_one"
    )

    # Calibrate noise
    sigma_g = accountant.calibrate_noise(
        epsilon_target=5.0,
        delta_target=1e-5
    )

    # Verify
    print("\nVerification:")
    results = accountant.verify_privacy(epsilon=5.0, delta=1e-5)
    for key, value in results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6e}")
        else:
            print(f"  {key}: {value}")