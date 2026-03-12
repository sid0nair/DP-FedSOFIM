import os
import json
import glob
import matplotlib.pyplot as plt


def plot_fl_comparison():
    # ---------------------------------------------------------------------
    # 1. Configure directories for the three algorithms
    # ---------------------------------------------------------------------
    alg_dirs = {
        "fedgd": "./fedgd_results",  # DP-FedGD sweep
        "sofim": "./sofim_results",
        "scaffold": "./scaffold_results"
    }

    # Colors keyed by epsilon
    eps_colors = {
        10: '#1f77b4',  # Solid (High budget)
        5: '#d62728',  # Dashed
        2: '#2ca02c',  # Dotted
        1: '#e349e0',  # Dash-dot specific sweep
        0.5: '#e3c034',# For very strict DP
        0:"#50c7c7"
    }

    # Line styles for the algorithms
    line_styles = {
        "fedgd": "--",  # Blue
        "sofim": "-",   # Red
        "scaffold": "-."
    }

    # Legend names
    alg_names = {
        "fedgd": "DP-FedGD",
        "sofim": "DP-FedSOFIM",
        "scaffold": "DP-Scaffold"
    }

    # ---------------------------------------------------------------------
    # 2. Collect ALL runs per (algorithm, epsilon)
    # ---------------------------------------------------------------------
    runs = {}

    for alg, results_dir in alg_dirs.items():
        if not os.path.exists(results_dir):
            print(f"Directory {results_dir} not found for {alg}, skipping.")
            continue

        pattern = os.path.join(results_dir, "*.json")
        files = sorted(glob.glob(pattern))
        print(f"[{alg}] Found {len(files)} result files.")

        for filepath in files:
            try:
                with open(filepath, "r") as f:
                    data = json.load(f)
            except Exception as e:
                print(f"❌ Error reading {filepath}: {e}")
                continue

            epsilon = None
            total_rounds = None
            sweep_cfg = data.get("sweep_config", {})

            if sweep_cfg:
                epsilon = sweep_cfg.get("epsilon", None)
                total_rounds = sweep_cfg.get("rounds", None)

            # Filename parsing fallback for fednew_eps{eps}_rounds{rounds}.json
            filename = os.path.basename(filepath).lower()
            if epsilon is None and "eps" in filename:
                try:
                    parts = filename.replace(".json", "").split("_")
                    for p in parts:
                        if p.startswith("eps"): epsilon = float(p.replace("eps", ""))
                        if p.startswith("rounds"): total_rounds = int(p.replace("rounds", ""))
                except Exception:
                    pass

            if epsilon is None or total_rounds is None:
                continue

            if float(epsilon).is_integer(): epsilon = int(epsilon)

            key = (alg, epsilon)
            runs.setdefault(key, []).append((filepath, total_rounds))

    # ---------------------------------------------------------------------
    # 3. Plot loss vs round (using the longest run for each (algorithm, epsilon))
    # ---------------------------------------------------------------------
    if not runs:
        print("No valid runs found. Nothing to plot.")
        return

    plt.figure(figsize=(12, 8))

    # Sort primarily by epsilon (to group line styles) and then algorithm
    # Plot order: all SOFIM first, then all FedGD; within each, eps high->low, with No-DP (eps=0) last
    alg_order = {"sofim": 0, "scaffold": 1, "fedgd": 2}

    def eps_sort_key(eps):
        # (False, -eps) sorts by descending eps; (True, 0) pushes eps=0 (No DP) to the end
        return (eps == 0, -float(eps))

    sorted_keys = sorted(
        runs.keys(),
        key=lambda k: (alg_order.get(k[0], 99), eps_sort_key(k[1]))
    )

    for (alg, epsilon) in sorted_keys:
        filelist = runs[(alg, epsilon)]

        # ---- pick the single longest run for this (alg, eps) ----
        filepath, total_rounds = max(filelist, key=lambda t: t[1])

        try:
            with open(filepath, "r") as f:
                data = json.load(f)

            rr = data.get("round_results", []) or data.get("results", {}).get("round_results", [])
            if not rr:
                continue

            loss_key_candidates = ["loss", "test_loss", "train_loss"]
            loss_key = next((k for k in loss_key_candidates if rr[0].get(k, None) is not None), None)
            if loss_key is None:
                continue

            losses = [r.get(loss_key) for r in rr if r.get(loss_key) is not None]
            if not losses:
                continue

            x = list(range(1, len(losses) + 1))

            if epsilon == 0:
                label = f"{alg_names.get(alg, alg)} (No DP)"
            else:
                label = f"{alg_names.get(alg, alg)} (ε={epsilon})"

            plt.plot(
                x, losses,
                label=label,
                color=eps_colors.get(epsilon, "gray"),
                linestyle=line_styles.get(alg, "-"),
                linewidth=2,
                alpha=0.85
            )

        except Exception as e:
            print(f"Skipping {filepath}: {e}")
            continue

    # ---------------------------------------------------------------------
    # 4. Final plot formatting
    # ---------------------------------------------------------------------
    plt.title("Second-Order FL Comparison: Loss vs Round", fontsize=14, fontweight='bold')
    plt.xlabel("Training Round", fontsize=12)
    plt.ylabel("Loss", fontsize=12)

    plt.grid(True, linestyle="--", alpha=0.6)

    # Place legend outside if it gets too crowded
    plt.legend(fontsize=9, loc='upper left', bbox_to_anchor=(1, 1))

    output_file = "fl_algorithm_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"✅ Comparison plot saved: {output_file}")
    plt.show()


if __name__ == "__main__":
    plot_fl_comparison()