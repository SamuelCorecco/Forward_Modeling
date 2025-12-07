import argparse
import yaml
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import torch

from src.data.loader import load_and_prepare_datasets

from src.models.wrappers import MultiResNetWrapper, FpcaXgbWrapper, StandardXgbWrapper, Paper2008MlpWrapper, PcaXgbWrapper
from src.models.components.resnet_emulator import StokesResNet, PhysicsLoss

class ConfigWrapper:
    def __init__(self, yaml_dict):
        self.DATASET_SAVE_PATH = yaml_dict['dataset']['path']
        self.SEED = yaml_dict['dataset']['seed']
        self.FIXED_SIZE = yaml_dict['dataset']['fixed_size']

def plot_reconstructions(y_true_dict, y_pred_full, indices, save_path, fixed_size):
    stokes = ['I', 'Q', 'U', 'V']
    n_samples = len(indices)
    
    wl_center = 6302.0
    wl_range = 1.5
    wl_axis = np.linspace(wl_center - wl_range, wl_center + wl_range, fixed_size)
    
    fig, axes = plt.subplots(n_samples, 4, figsize=(24, 4 * n_samples), constrained_layout=True)
    
    if n_samples == 1: axes = axes.reshape(1, -1)

    preds_separated = {
        'I': y_pred_full[:, 0:fixed_size],
        'Q': y_pred_full[:, fixed_size:2*fixed_size],
        'U': y_pred_full[:, 2*fixed_size:3*fixed_size],
        'V': y_pred_full[:, 3*fixed_size:4*fixed_size]
    }

    for row_idx, sample_idx in enumerate(indices):
        for col_idx, s in enumerate(stokes):
            ax = axes[row_idx, col_idx]
            
            real_profile = y_true_dict[s][sample_idx]
            pred_profile = preds_separated[s][sample_idx]
            
            # --- 2. Plot con Asse X Fisico ---
            ax.plot(wl_axis, real_profile, label='Real', color='black', linewidth=1.5, alpha=0.7)
            ax.plot(wl_axis, pred_profile, label='Pred', color='red', linestyle='--', linewidth=1.5)
            
            if row_idx == 0:
                ax.set_title(f"Stokes {s}", fontsize=16)
            
            
            if row_idx == n_samples - 1:
                ax.set_xlabel("Wavelength [Å]", fontsize=12)
                
            ax.grid(True, alpha=0.3, linestyle=':')
            
            if s in ['Q', 'U']:
                y_max = max(np.max(np.abs(real_profile)), np.max(np.abs(pred_profile)))
                if y_max > 0:
                    ax.set_ylim(-y_max*1.3, y_max*1.3)
            
            if row_idx == 0 and col_idx == 0:
                ax.legend(loc='upper right', frameon=True)
    
    print(f"Saving plot to {save_path}...")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config used for training")
    parser.add_argument("--indices", type=int, nargs='+', default=[0, 42, 100, 150], help="List of test set indices to plot")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    cfg = ConfigWrapper(config_dict)
    
    exp_name = config_dict['experiment_name']
    results_dir = Path(config_dict['output_dir']) / exp_name
    model_path = results_dir / "model.pkl"
    
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        return

    print("Loading data...")
    _, X_test, _, y_test_dict = load_and_prepare_datasets(cfg)
    
    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)
    
    if hasattr(model, 'device'):
        device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
        model.device = device
        if hasattr(model, 'models'):
            for sub_model in model.models.values():
                sub_model.to(device)

    indices = args.indices
    max_idx = len(X_test) - 1
    indices = [i for i in indices if i <= max_idx]
    
    print(f"Predicting for indices: {indices}...")
    X_subset = X_test.iloc[indices]
    
    y_pred_full = model.predict(X_subset)
    
    save_plot_path = results_dir / "reconstruction_examples.png"
    
    y_true_subset = {
        s: y_test_dict[s][indices] for s in ['I', 'Q', 'U', 'V']
    }
    
    indices_relative = list(range(len(indices)))
    
    plot_reconstructions(
        y_true_subset, 
        y_pred_full, 
        indices_relative,
        save_plot_path, 
        cfg.FIXED_SIZE
    )
    
    print("Done.")

if __name__ == "__main__":
    main()