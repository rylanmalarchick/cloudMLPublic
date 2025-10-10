import os
import json
import pandas as pd
from glob import glob


def aggregate_ablation_results(
    results_dir="results", output_file="ablation_summary.csv"
):
    """
    Aggregates metrics from ablation runs stored as JSON files in results_dir.
    Assumes files like 'loo_fold_{flight}_metrics.json' or 'metrics_{run_name}.json'.
    Outputs a CSV with one row per run/flight.
    """
    results = []

    # Find all JSON metric files
    json_files = glob(os.path.join(results_dir, "*_metrics.json"))

    for file_path in json_files:
        with open(file_path, "r") as f:
            metrics = json.load(f)

        # Extract run info from filename
        filename = os.path.basename(file_path)
        if "loo_fold" in filename:
            flight = filename.split("_")[2]
            run_type = "LOO"
        else:
            flight = "N/A"
            run_type = filename.replace("_metrics.json", "").replace("metrics_", "")

        # Add metadata
        metrics["run_type"] = run_type
        metrics["flight"] = flight
        metrics["file"] = filename

        results.append(metrics)

    if not results:
        print("No metric files found.")
        return

    # Create DataFrame and save
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"Aggregated results saved to {output_file}")
    print(df.head())


if __name__ == "__main__":
    aggregate_ablation_results()
