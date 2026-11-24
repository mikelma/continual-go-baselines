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
import re


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


@app.command
def latex_table(
    csv_paths: list[str],
    output_path: str | None = None,
    decimals: int = 2,
    bold_best: bool = True,
    normalize_by: float = 1e6,
):
    """Generate a booktabs LaTeX table comparing PPO configurations across benchmarks."""

    config_cols = ["weight_decay", "actor_lr", "layer_norm"]

    def extract_benchmark_name(run_names: list[str]) -> str:
        """Extract the benchmark prefix from run names."""
        benchmark_names = set()

        for run_name in run_names:
            name = re.sub(r"[-_]seed\d+.*$", "", run_name)
            name = re.sub(r"[-_]rep\d+$", "", name)
            benchmark_names.add(name)

        if len(benchmark_names) != 1:
            raise ValueError(
                "Could not infer a unique benchmark name from run_name values. "
                f"Extracted names: {sorted(benchmark_names)}. "
                "Expected names such as '9x9-k16-1-seed42-rep0'."
            )

        return benchmark_names.pop()

    def calculate_auc(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate mean and std AUC across runs for each configuration."""

        all_steps = sorted(df["env_steps"].dropna().unique())

        if len(all_steps) == 0:
            raise ValueError("No valid 'env_steps' found in the dataset.")

        common_x = np.insert(all_steps, 0, 0.0)

        run_aucs = []

        for keys, group in df.groupby(config_cols + ["run_name"]):
            wd, lr, ln, run_name = keys

            group = group.sort_values("env_steps")
            x = group["env_steps"].values
            y = group["avg_reward_per_step"].values

            if len(x) > 1:
                interpolated_y = np.interp(
                    all_steps,
                    x,
                    y,
                    left=y[0],
                    right=y[-1],
                )
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
                    "auc": auc_value / normalize_by,
                }
            )

        if not run_aucs:
            raise ValueError("No valid runs found in the dataset.")

        run_auc_df = pd.DataFrame(run_aucs)

        grouped = (
            run_auc_df.groupby(config_cols)["auc"]
            .agg(mean="mean", std="std")
            .reset_index()
        )

        grouped["std"] = grouped["std"].fillna(0.0)

        return grouped

    def latex_escape(value) -> str:
        """Escape special LaTeX characters."""
        return (
            str(value)
            .replace("\\", r"\textbackslash{}")
            .replace("&", r"\&")
            .replace("%", r"\%")
            .replace("_", r"\_")
            .replace("#", r"\#")
            .replace("{", r"\{")
            .replace("}", r"\}")
        )

    def format_config_value(value) -> str:
        """Format hyperparameters, using scientific notation when appropriate."""
        if isinstance(value, (bool, np.bool_)):
            return str(value)

        if isinstance(value, (int, np.integer)):
            return str(value)

        if isinstance(value, (float, np.floating)):
            if value == 0:
                return "0"

            exponent = int(np.floor(np.log10(abs(value))))

            if exponent <= -3 or exponent >= 4:
                mantissa = value / (10**exponent)
                return rf"${mantissa:.2g} \times 10^{{{exponent}}}$"

            return f"{value:g}"

        return latex_escape(value)

    # ---------------------------------------------------------
    # Load each CSV and calculate its benchmark-specific AUCs.
    # ---------------------------------------------------------

    benchmark_results = {}

    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)

        required_cols = set(
            config_cols + ["env_steps", "avg_reward_per_step", "run_name"]
        )
        missing_cols = required_cols - set(df.columns)

        if missing_cols:
            raise ValueError(
                f"{csv_path} is missing required columns: {sorted(missing_cols)}"
            )

        run_names = df["run_name"].dropna().astype(str).unique().tolist()

        if not run_names:
            raise ValueError(f"No valid run_name values found in {csv_path}.")

        benchmark = extract_benchmark_name(run_names)

        if benchmark in benchmark_results:
            raise ValueError(
                f"Multiple CSV files correspond to benchmark '{benchmark}'."
            )

        print(f"Processing benchmark: {benchmark} ({csv_path})")

        grouped = calculate_auc(df)

        # Store results indexed by PPO configuration.
        stats = {}

        for _, row in grouped.iterrows():
            config = tuple(row[col] for col in config_cols)
            stats[config] = (row["mean"], row["std"])

        benchmark_results[benchmark] = stats

    # ---------------------------------------------------------
    # Get the union of all PPO configurations.
    # ---------------------------------------------------------

    all_config_rows = []

    for stats in benchmark_results.values():
        all_config_rows.extend(stats.keys())

    configs_df = pd.DataFrame(all_config_rows, columns=config_cols)
    configs_df = configs_df.drop_duplicates().sort_values(config_cols)

    configs = list(configs_df.itertuples(index=False, name=None))

    benchmarks = list(benchmark_results.keys())
    # ---------------------------------------------------------
    # Generate the LaTeX table.
    # ---------------------------------------------------------

    num_benchmarks = len(benchmarks)
    num_columns = 3 + num_benchmarks

    column_spec = r"@{\extracolsep{\fill}}lll" + "c" * num_benchmarks + r"@{}"

    # Find the best and second-best distinct means per benchmark.
    benchmark_ranks = {}

    for benchmark, stats in benchmark_results.items():
        means = sorted(
            [mean for mean, std in stats.values() if np.isfinite(mean)],
            reverse=True,
        )

        # Distinct values, treating numerically close values as ties.
        distinct_means = []
        for mean in means:
            if not any(np.isclose(mean, existing) for existing in distinct_means):
                distinct_means.append(mean)

        benchmark_ranks[benchmark] = {
            "best": distinct_means[0] if len(distinct_means) >= 1 else None,
            "second": distinct_means[1] if len(distinct_means) >= 2 else None,
        }

    # Count rows for multirow grouping.
    wd_counts = configs_df.groupby("weight_decay", sort=False).size().to_dict()

    lr_counts = (
        configs_df.groupby(
            ["weight_decay", "actor_lr"],
            sort=False,
        )
        .size()
        .to_dict()
    )

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{6pt}",
        r"\renewcommand{\arraystretch}{1.2}",
        (
            r"\caption{Normalized AUC comparison across PPO configurations "
            r"and benchmarks. Results are reported as mean $\pm$ standard "
            r"deviation across runs. Best results are shown in bold and "
            r"second-best results are underlined.}"
        ),
        r"\label{tab:ppo_auc_benchmarks}",
        rf"\begin{{tabular*}}{{\textwidth}}{{{column_spec}}}",
        r"\toprule",
        (
            r"\multicolumn{3}{c}{PPO Configuration} & "
            rf"\multicolumn{{{num_benchmarks}}}{{c}}"
            r"{Normalized AUC ($\uparrow$)} \\"
        ),
        rf"\cmidrule(lr){{1-3}}\cmidrule(lr){{4-{num_columns}}}",
        (
            r"Weight Decay & Actor LR & Layer Norm & "
            + " & ".join(latex_escape(b) for b in benchmarks)
            + r" \\"
        ),
        r"\midrule",
    ]

    previous_wd = None
    previous_lr = None

    for i, config in enumerate(configs):
        wd, lr, ln = config

        new_wd = previous_wd is None or wd != previous_wd
        new_lr = new_wd or lr != previous_lr

        # Separate weight-decay groups and actor-LR subgroups.
        if i > 0:
            if new_wd:
                lines.append(r"\midrule")
            elif new_lr:
                lines.append(rf"\cmidrule(lr){{3-{num_columns}}}")

        # Merge repeated weight-decay values.
        if new_wd:
            wd_cell = (
                rf"\multirow{{{wd_counts[wd]}}}{{*}}"
                rf"{{{format_config_value(wd)}}}"
            )
        else:
            wd_cell = ""

        # Merge repeated actor-LR values within each weight decay.
        if new_lr:
            lr_cell = (
                rf"\multirow{{{lr_counts[(wd, lr)]}}}{{*}}"
                rf"{{{format_config_value(lr)}}}"
            )
        else:
            lr_cell = ""

        cells = [
            wd_cell,
            lr_cell,
            format_config_value(ln),
        ]

        for benchmark in benchmarks:
            stats = benchmark_results[benchmark]

            if config not in stats:
                cells.append("--")
                continue

            mean, std = stats[config]

            mean_str = f"{mean:.{decimals}f}"
            std_str = f"{std:.{decimals}f}"

            best = benchmark_ranks[benchmark]["best"]
            second = benchmark_ranks[benchmark]["second"]

            if bold_best and best is not None and np.isclose(mean, best):
                cell = rf"$\boldsymbol{{{mean_str} \pm {std_str}}}$"

            elif second is not None and np.isclose(mean, second):
                cell = rf"$\underline{{{mean_str} \pm {std_str}}}$"

            else:
                cell = rf"${mean_str} \pm {std_str}$"

            cells.append(cell)

        lines.append(" & ".join(cells) + r" \\")

        previous_wd = wd
        previous_lr = lr

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular*}",
            r"\end{table*}",
        ]
    )

    latex = "\n".join(lines)

    if output_path is not None:
        with open(output_path, "w") as f:
            f.write(latex + "\n")

        print(f"LaTeX table saved to: {output_path}")
    else:
        print(latex)


if __name__ == "__main__":
    app.cli()
