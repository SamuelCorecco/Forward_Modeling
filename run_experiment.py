import argparse
import yaml
import time
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from src.data.loader import load_and_prepare_datasets
from src.evaluation import (
    rmse_score, derivative_correlation, 
    concordance_correlation_coefficient, peak_amplitude_error
)

from src.model_builder import build_model

class ConfigWrapper:
    def __init__(self, yaml_dict):
        self.DATASET_SAVE_PATH = yaml_dict['dataset']['path']
        self.SEED = yaml_dict['dataset']['seed']
        self.FIXED_SIZE = yaml_dict['dataset']['fixed_size']
        self.exp_name = yaml_dict['experiment_name']
        self.out_dir = yaml_dict['output_dir']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    print(f"Loading configuration from {args.config}...")
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    cfg = ConfigWrapper(config_dict)

    exp_dir = Path(cfg.out_dir) / cfg.exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train_dict, y_test_dict = load_and_prepare_datasets(cfg)
    
    y_train_full = np.hstack([y_train_dict[s] for s in ['I', 'Q', 'U', 'V']])

    print(f"\n--- Building Model: {config_dict.get('model_type', 'Unknown')} ---")
    model = build_model(config_dict)

    print(f"Starting training on {len(X_train)} samples...")
    t0 = time.time()
    
    model.fit(X_train, y_train_full)
    
    train_time_sec = time.time() - t0
    print(f"Training completed in {train_time_sec:.2f}s")

    print("\n--- Prediction & Benchmarking ---")
    print(f"Starting evaluation on {len(X_test)} samples...")
    t0 = time.time()
    y_pred_full = model.predict(X_test)
    total_pred_time = time.time() - t0
    throughput_ms = (total_pred_time / len(X_test)) * 1000
    
    X_single = X_test.iloc[[0]]
    t_lat = time.time()
    for _ in range(50): _ = model.predict(X_single)
    single_latency_ms = ((time.time() - t_lat) / 50) * 1000
    
    print(f"Throughput: {throughput_ms:.4f} ms/profile | Latency: {single_latency_ms:.4f} ms")

    print("\n--- Calculating Physics Metrics ---")
    y_pred_dict = {
        'I': y_pred_full[:, 0:cfg.FIXED_SIZE],
        'Q': y_pred_full[:, cfg.FIXED_SIZE:2*cfg.FIXED_SIZE],
        'U': y_pred_full[:, 2*cfg.FIXED_SIZE:3*cfg.FIXED_SIZE],
        'V': y_pred_full[:, 3*cfg.FIXED_SIZE:4*cfg.FIXED_SIZE]
    }

    metrics_rows = []
    for s in ['I', 'Q', 'U', 'V']:
        yt, yp = y_test_dict[s], y_pred_dict[s]
        metrics_rows.append({
            "Stokes": s,
            "RMSE": rmse_score(yt, yp),
            "CCC": concordance_correlation_coefficient(yt, yp),
            "Peak Err %": peak_amplitude_error(yt, yp),
            "Deriv Corr": derivative_correlation(yt, yp, stokes_type=s)
        })

    df_metrics = pd.DataFrame(metrics_rows)
    df_metrics.to_csv(exp_dir / "metrics.csv", index=False)
    print("\n" + df_metrics.round(4).to_string(index=False))

    pd.DataFrame({
        "train_time_s": [train_time_sec],
        "throughput_ms": [throughput_ms],
        "latency_ms": [single_latency_ms]
    }).to_csv(exp_dir / "performance.csv", index=False)

    joblib.dump(model, exp_dir / "model.pkl")
    with open(exp_dir / "config_snapshot.yaml", 'w') as f:
        yaml.dump(config_dict, f)
    
    print(f"\nExperiment finished. Results in {exp_dir}")

if __name__ == "__main__":
    main()