# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "matplotlib",
#     "pandas",
#     "wandb",
#     "tyro",
# ]
# ///

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import pandas as pd
import tyro
import wandb
import pprint


@dataclass
class Args:
    out_csv: str = "wandb.csv"
    entity: str = "mmalagon"
    project: str = "continual-go-ppo"
    benchmark: str = "9x9-k16-A"
    hours_ago: float = 24.0  # Number of hours back to look for runs
    max_workers: int = 16  # Concurrency threads


def main(args: Args):
    api = wandb.Api()

    time_threshold = (
        datetime.now(timezone.utc) - timedelta(hours=args.hours_ago)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")

    filters = {
        "$and": [
            {"created_at": {"$gt": time_threshold}},
            {"state": {"$in": ["finished", "running"]}},
            {"config.benchmark": args.benchmark},
        ]
    }
    print("Filters are:")
    pprint.pprint(filters)
    runs = list(api.runs(f"{args.entity}/{args.project}", filters=filters))

    print(f"Found {len(runs)} runs to download. Starting parallel fetch...")

    def fetch_run_history(run):
        """Helper function to fetch history for a single run in a worker thread."""
        try:
            print(f"Fetching: {run.name}")
            hist = run.history(keys=["env_steps", "avg_reward_per_step"])

            hist["run_name"] = run.name
            hist["weight_decay"] = run.config.get("weight_decay")
            hist["actor_lr"] = run.config.get("actor_lr")
            hist["layer_norm"] = run.config.get("layer_norm")
            hist["created_at"] = run.created_at
            return hist
        except Exception as e:
            print(f"Error fetching {run.name}: {e}")
            return None

    all_histories = []

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(fetch_run_history, run): run for run in runs}
        for future in as_completed(futures):
            hist = future.result()
            if hist is not None and not hist.empty:
                all_histories.append(hist)

    if all_histories:
        master_df = pd.concat(all_histories, ignore_index=True)
        master_df.to_csv(args.out_csv, index=False)
        print(f"Successfully exported {len(all_histories)} runs to {args.out_csv}!")
    else:
        print("No runs found matching the criteria.")


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
