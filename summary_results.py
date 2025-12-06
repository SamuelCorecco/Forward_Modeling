import pandas as pd
from pathlib import Path
from tabulate import tabulate
import warnings

warnings.filterwarnings("ignore")

def load_results(results_dir="results"):
    root = Path(results_dir)

    rmse_rows = []
    ccc_rows = []
    peak_rows = []
    deriv_rows = []
    perf_rows = []

    for exp_dir in root.iterdir():
        if not exp_dir.is_dir():
            continue

        exp_name = exp_dir.name

        metrics_path = exp_dir / "metrics.csv"
        if metrics_path.exists():
            df_m = pd.read_csv(metrics_path)
            df_m = df_m.set_index("Stokes").loc[["I", "Q", "U", "V"]]

            rmse_rows.append({
                "Experiment": exp_name,
                "I": df_m.loc["I", "RMSE"],
                "Q": df_m.loc["Q", "RMSE"],
                "U": df_m.loc["U", "RMSE"],
                "V": df_m.loc["V", "RMSE"],
                "MEAN": df_m["RMSE"].mean()
            })

            ccc_rows.append({
                "Experiment": exp_name,
                "I": df_m.loc["I", "CCC"],
                "Q": df_m.loc["Q", "CCC"],
                "U": df_m.loc["U", "CCC"],
                "V": df_m.loc["V", "CCC"],
                "MEAN": df_m["CCC"].mean()
            })

            peak_rows.append({
                "Experiment": exp_name,
                "I": df_m.loc["I", "Peak Err %"],
                "Q": df_m.loc["Q", "Peak Err %"],
                "U": df_m.loc["U", "Peak Err %"],
                "V": df_m.loc["V", "Peak Err %"],
                "MEAN": df_m["Peak Err %"].mean()
            })

            deriv_rows.append({
                "Experiment": exp_name,
                "I": df_m.loc["I", "Deriv Corr"],
                "Q": df_m.loc["Q", "Deriv Corr"],
                "U": df_m.loc["U", "Deriv Corr"],
                "V": df_m.loc["V", "Deriv Corr"],
                "MEAN": df_m["Deriv Corr"].mean()
            })

        perf_path = exp_dir / "performance.csv"
        if perf_path.exists():
            df_p = pd.read_csv(perf_path).iloc[0]
            perf_rows.append({
                "Experiment": exp_name,
                "train_time_s": df_p["train_time_s"],
                "throughput_ms": df_p["throughput_ms"],
                "latency_ms": df_p["latency_ms"]
            })

    df_rmse   = pd.DataFrame(rmse_rows).round(5)
    df_ccc    = pd.DataFrame(ccc_rows).round(5)
    df_peak   = pd.DataFrame(peak_rows).round(5)
    df_deriv  = pd.DataFrame(deriv_rows).round(5)
    df_perf   = pd.DataFrame(perf_rows).round(5)

    return df_rmse, df_ccc, df_peak, df_deriv, df_perf


def print_table(title, df):
    print(f"\n{title}")
    print(tabulate(df, headers="keys", tablefmt="github", showindex=False))
    print()



df_rmse, df_ccc, df_peak, df_deriv, df_perf = load_results("results")

print_table("RMSE TABLE", df_rmse)
print_table("CCC TABLE", df_ccc)
print_table("PEAK ERROR TABLE", df_peak)
print_table("DERIVATIVE CORRELATION TABLE", df_deriv)
print_table("PERFORMANCE TABLE", df_perf)
