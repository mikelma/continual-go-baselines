# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "matplotlib",
#     "pandas",
#     "tyro",
# ]
# ///

import matplotlib.pyplot as plt
import pandas as pd
import tyro
from tyro.extras import SubcommandApp
import numpy as np


app = SubcommandApp()


@app.command
def line_plot(csv_path: str = "wandb.csv", palette: str = "tab10"):
    df = pd.read_csv(csv_path)

    config_cols = ["weight_decay", "actor_lr", "layer_norm"]
    grouped = (
        df.groupby(config_cols + ["env_steps"])["avg_reward_per_step"]
        .agg(["mean"])
        .reset_index()
    )

    unique_wds = sorted(df["weight_decay"].unique())
    unique_lrs = sorted(df["actor_lr"].unique())
    unique_lns = sorted(df["layer_norm"].unique(), key=str)

    unique_wds = sorted(df["weight_decay"].unique())
    cmap = plt.get_cmap(palette)
    wd_colors = {wd: cmap(i) for i, wd in enumerate(unique_wds)}

    linestyles = ["-", "--", "-.", ":"]
    ln_styles = {ln: linestyles[i % len(linestyles)] for i, ln in enumerate(unique_lns)}

    markers = ["o", "s", "^", "D", "v", "p", "*"]
    lr_markers = {lr: markers[i % len(markers)] for i, lr in enumerate(unique_lrs)}

    plt.figure(figsize=(14, 8))

    for keys, group in grouped.groupby(config_cols):
        wd, lr, ln = keys
        label = f"WD: {wd} | LR: {lr} | LN: {ln}"

        group = group.sort_values("env_steps")
        x = group["env_steps"]
        y_mean = group["mean"]

        color = wd_colors.get(wd, "#333333")
        linestyle = ln_styles.get(ln, "-")
        marker = lr_markers.get(lr, None)

        plt.plot(
            x,
            y_mean,
            color=color,
            linestyle=linestyle,
            marker=marker,
            markevery=max(1, len(x) // 12),
            linewidth=2,
            markersize=9,
            markeredgecolor="white",
            label=label,
        )

    plt.xlabel("Environment steps", fontsize=12)
    plt.ylabel("Average reward per step", fontsize=12)
    plt.title(
        "Performance Comparison Across Hyperparameter Configurations\n(Color ="
        " Weight Decay | Line Style = Layer Norm | Marker = Actor LR)",
        fontsize=13,
        # fontweight="bold",
        pad=15,
    )

    plt.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize="small",
        frameon=True,
        handlelength=4.0,
    )
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    plt.show()


@app.command
def bar_plot(csv_path: str = "wandb.csv", palette: str = "tab10"):
    df = pd.read_csv(csv_path)

    config_cols = ["weight_decay", "actor_lr", "layer_norm"]

    # Define a common grid of environment steps for fair AUC comparison across runs
    all_steps = sorted(df["env_steps"].unique())
    if len(all_steps) == 0:
        raise ValueError("No 'env_steps' found in the dataset.")

    common_x = np.insert(all_steps, 0, 0.0)

    # Calculate AUC for each individual run/seed using a shared step grid via interpolation
    run_aucs = []
    for keys, group in df.groupby(config_cols + ["run_name"]):
        wd, lr, ln, run_name = keys
        group = group.sort_values("env_steps")
        x = group["env_steps"].values
        y = group["avg_reward_per_step"].values

        if len(x) > 1:
            # Interpolate y-values onto the common step grid (with 0.0 starting value)
            # Extrapolate or fill missing bounds safely (e.g., filling initial step with y[0])
            interpolated_y = np.interp(all_steps, x, y, left=y[0], right=y[-1])
            full_y = np.insert(interpolated_y, 0, y[0])
            auc_value = np.trapezoid(full_y, common_x)
        else:
            auc_value = 0.0

        run_aucs.append(
            {
                "weight_decay": wd,
                "actor_lr": lr,
                "layer_norm": ln,
                "run_name": run_name,
                "auc": auc_value,
            }
        )

    run_auc_df = pd.DataFrame(run_aucs)

    # Aggregate mean and std across runs for each configuration
    grouped = (
        run_auc_df.groupby(config_cols)["auc"].agg(mean="mean", std="std").reset_index()
    )
    grouped["std"] = grouped["std"].fillna(0.0)

    unique_wds = sorted(run_auc_df["weight_decay"].unique())
    unique_lns = sorted(run_auc_df["layer_norm"].unique(), key=str)

    cmap = plt.get_cmap(palette)
    wd_colors = {wd: cmap(i) for i, wd in enumerate(unique_wds)}

    # Define textures (hatches) for distinguishing categories (layer norm)
    hatches = ["", "/", "\\", "x", ".", "*", "o", "-"]
    ln_hatches = {ln: hatches[i % len(hatches)] for i, ln in enumerate(unique_lns)}

    configs, means, stds, colors, bar_hatches = [], [], [], [], []

    for _, row in grouped.iterrows():
        wd = row["weight_decay"]
        lr = row["actor_lr"]
        ln = row["layer_norm"]

        configs.append(f"WD:{wd}\nLR:{lr}\nLN:{ln}")
        means.append(row["mean"])
        stds.append(row["std"])
        colors.append(wd_colors.get(wd, "gray"))
        bar_hatches.append(ln_hatches.get(ln, ""))

    plt.figure(figsize=(14, 6))
    bars = plt.bar(
        configs,
        means,
        yerr=stds,
        capsize=4,
        color=colors,
        edgecolor="black",
        linewidth=0.8,
        alpha=0.85,
    )

    # Apply textures (hatches) to individual bars
    for bar, hatch in zip(bars, bar_hatches):
        bar.set_hatch(hatch)

    plt.xlabel("Hyperparameter Configuration", fontsize=12)
    plt.ylabel("Area Under Curve (AUC)", fontsize=12)
    plt.title(
        "Area Under the Curve (AUC) by Configuration with Standard Deviation (Reference at x=0)",
        fontsize=13,
        pad=15,
    )
    plt.xticks(rotation=0, fontsize=9)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    app.cli()
