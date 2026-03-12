import os
import json
import glob
import matplotlib.pyplot as plt


def plot_fl_comparison():
    alg_dirs = {
        "fedgd": "./fedgd_clients100_path",
        "sofim": "./sofim_clients100_path",
        "fedfc": "./fedfc_clients100_path",
        "scaffold": "./scaffold_clients100_path"
    }

    alg_linestyles = {
        "fedgd": "-.",
        "sofim": "-",
        "fedfc": "--",
        "scaffold": ":"
    }

    eps_colors = {
        10: '#1f77b4',  # Solid (High budget)
        5: '#d62728',  # Dashed
        2: '#2ca02c',  # Dotted
        1: '#e349e0',  # Dash-dot specific sweep
        0.5: '#e3c034',  # For very strict DP
        0: "#50c7c7"
    }

    alg_names = {
        "fedgd": "DP-FedGD",
        "sofim": "DP-FedSOFIM",
        "fedfc": "DP-FedFC",
        "scaffold": "DP-SCAFFOLD"
    }

    # Collect data: (alg, epsilon) -> [(round_num, accuracy), ...]
    runs = {}

    for alg, results_dir in alg_dirs.items():
        if not os.path.exists(results_dir):
            print(f"Directory {results_dir} not found, skipping.")
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

            # Get epsilon
            epsilon = None
            sweep_cfg = data.get("sweep_config", {})
            config = data.get("config", {})

            epsilon = sweep_cfg.get("epsilon") or config.get("epsilon")

            # Fallback to filename
            if epsilon is None:
                filename = os.path.basename(filepath).lower()
                if "eps" in filename:
                    try:
                        parts = filename.replace(".json", "").split("_")
                        for p in parts:
                            if p.startswith("eps"):
                                epsilon = float(p.replace("eps", ""))
                                break
                    except Exception:
                        pass

            if epsilon is None:
                print(f"⚠️  No epsilon found for {os.path.basename(filepath)}, skipping")
                continue

            if float(epsilon).is_integer():
                epsilon = int(epsilon)

            # Get the FINAL round number this file trained to
            total_rounds = sweep_cfg.get("rounds") or config.get("federated_rounds")

            # Fallback to filename
            if total_rounds is None:
                filename = os.path.basename(filepath).lower()
                if "rounds" in filename:
                    try:
                        parts = filename.replace(".json", "").split("_")
                        for p in parts:
                            if p.startswith("rounds"):
                                total_rounds = int(p.replace("rounds", ""))
                                break
                    except Exception:
                        pass

            if total_rounds is None:
                print(f"⚠️  No rounds found for {os.path.basename(filepath)}, skipping")
                continue

            # Get the FINAL accuracy from final_stats
            final_stats = data.get("final_stats", {})
            accuracy = final_stats.get("accuracy")

            if accuracy is None:
                print(f"⚠️  No final accuracy in {os.path.basename(filepath)}, skipping")
                continue

            # Store: (round_number, accuracy)
            key = (alg, epsilon)
            runs.setdefault(key, []).append((total_rounds, accuracy))

            print(f"  {os.path.basename(filepath)}: Round {total_rounds} → Acc {accuracy:.4f}")

    if not runs:
        print("❌ No valid data found.")
        return

    # Plot
    fig, ax = plt.subplots(figsize=(20, 9))

    alg_order = {"sofim": 0, "fedgd": 1, "fedfc": 2, "scaffold": 3}

    def eps_sort_key(eps):
        return (eps == 0, -float(eps))

    sorted_keys = sorted(
        runs.keys(),
        key=lambda k: (alg_order.get(k[0], 99), eps_sort_key(k[1]))
    )

    print("\n" + "=" * 70)
    print("PLOTTING DATA")
    print("=" * 70)

    for (alg, epsilon) in sorted_keys:
        data_points = runs[(alg, epsilon)]

        # Sort by round number
        data_points.sort(key=lambda x: x[0])

        xs = [round_num for round_num, _ in data_points]
        ys = [acc for _, acc in data_points]

        print(f"\n{alg_names.get(alg, alg)} (ε={epsilon}):")
        print(f"  Points: {list(zip(xs, [f'{y:.4f}' for y in ys]))}")

        label = f"{alg_names.get(alg, alg)} (ε={epsilon})" if epsilon != 0 else f"{alg_names.get(alg, alg)} (No DP)"

        # Plot line
        ax.plot(
            xs, ys,
            label=label,
            color=eps_colors.get(epsilon, "gray"),
            linestyle=alg_linestyles.get(alg, "-"),
            linewidth=3,
            alpha=0.8
        )

        # Plot markers
        # Plot markers - squares for SOFIM, circles for others
        marker_style = 's' if alg == 'sofim' else 'o'
        ax.scatter(
            xs, ys,
            color=eps_colors.get(epsilon, "gray"),  # Also fix: should be epsilon, not alg
            s=160,
            alpha=0.9,
            edgecolors='white',
            linewidths=2,
            zorder=5,
            marker=marker_style
        )

    ax.set_title("DP-FedSOFIM vs DP-FedGD vs DP-FedFC vs DP-SCAFFOLD on PathMNIST\n(100 clients)",
                 fontsize=22, fontweight='bold', pad=20)
    ax.set_xlabel("Federated Rounds", fontsize=20, fontweight='bold')
    ax.set_ylabel("Test Accuracy", fontsize=20, fontweight='bold')

    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0.05, top=0.75)

    ax.grid(True, linestyle="--", alpha=0.4, linewidth=1)
    ax.set_axisbelow(True)

    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.set_xlim(left=0, right=75)
    ax.legend(fontsize=13, loc="center left", bbox_to_anchor=(1.01, 0.5),
              framealpha=0.95, edgecolor='black', fancybox=True, shadow=True,
              ncol=1, handlelength=3, handleheight=1.5)

    plt.tight_layout()

    output_file = "fl_comparison_acc.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight", pad_inches=0.5)

    print(f"\n{'=' * 70}")
    print(f"✅ Plot saved: {output_file}")
    print(f"{'=' * 70}\n")

    plt.show()


if __name__ == "__main__":
    plot_fl_comparison()