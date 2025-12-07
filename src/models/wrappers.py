import xgboost as xgb
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.multioutput import MultiOutputRegressor
import time

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import numpy as np


from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor

import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
import torch.optim as optim
from src.models.components.resnet_emulator import StokesResNet, PhysicsLoss

import torch.nn as nn
from tqdm import tqdm
import copy

class FpcaXgbWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, fpca_transformer, xgb_params):
        self.fpca_transformer = fpca_transformer
        self.xgb_params = xgb_params
        self.model = MultiOutputRegressor(xgb.XGBRegressor(**xgb_params), n_jobs=-1)
        
    def fit(self, X, y):
        print("Fitting fPCA on target profiles...")
        y_coeffs = self.fpca_transformer.fit_transform(y)
        
        print(f"Training XGBoost on {y_coeffs.shape[1]} coefficients...")
        start = time.time()
        self.model.fit(X, y_coeffs)
        print(f"XGBoost training done in {time.time() - start:.2f}s")
        
        return self

    def predict(self, X):
        y_coeffs_pred = self.model.predict(X)
        
        y_profiles_pred = self.fpca_transformer.inverse_transform(y_coeffs_pred)
        
        return y_profiles_pred
    



class Paper2008MlpWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, pca_transformer, mlp_params, arch_iv, arch_qu):
        self.pca_transformer = pca_transformer
        self.mlp_params = mlp_params
        self.arch_iv = arch_iv 
        self.arch_qu = arch_qu 
        
        self.models = {}
        self.scaler_X = StandardScaler()
        self.stokes_list = ['I', 'Q', 'U', 'V']

    def fit(self, X, y):
        X_scaled = self.scaler_X.fit_transform(X)
        
        y_coeffs_all = self.pca_transformer.fit_transform(y)
        
        idx_start = 0
        for s in self.stokes_list:
            n_comps = self.pca_transformer.n_components_kept[s]
            y_s = y_coeffs_all[:, idx_start : idx_start + n_comps]
            idx_start += n_comps
            
            hidden_layers = self.arch_iv if s in ['I', 'V'] else self.arch_qu
            
            print(f"Training MLP for Stokes {s} | Arch: {hidden_layers} | Input Feat: {X.shape[1]} -> Out Coeffs: {n_comps}")
            
            mlp = MLPRegressor(hidden_layer_sizes=hidden_layers, **self.mlp_params)
            mlp.fit(X_scaled, y_s)
            self.models[s] = mlp
            
        return self

    def predict(self, X):
        X_scaled = self.scaler_X.transform(X)
        
        coeffs_preds = []
        for s in self.stokes_list:
            pred_s = self.models[s].predict(X_scaled)
            if pred_s.ndim == 1: pred_s = pred_s.reshape(-1, 1)
            coeffs_preds.append(pred_s)
            
        y_coeffs_pred_full = np.hstack(coeffs_preds)
        
        return self.pca_transformer.inverse_transform(y_coeffs_pred_full)
    



class StandardXgbWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, xgb_params):
        self.xgb_params = xgb_params
        self.model = MultiOutputRegressor(XGBRegressor(**xgb_params), n_jobs=-1)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)
    
class MultiResNetWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, 
                 hidden_dim=256, 
                 n_blocks=4, 
                 lr=1e-3, 
                 batch_size=256, 
                 epochs=100, 
                 deriv_weight=1.0, 
                 patience=10, 
                 random_state=42, 
                 device=None):
        
        self.hidden_dim = hidden_dim
        self.n_blocks = n_blocks
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.deriv_weight = deriv_weight
        self.patience = patience
        self.random_state = random_state
        
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
            
        self.models = {}
        self.scalers_y = {} 
        self.scaler_x = StandardScaler()
        self.stokes_labels = ['I', 'Q', 'U', 'V']

    def fit(self, X, y):
        if hasattr(X, "values"): X = X.values
        
        X_scaled = self.scaler_x.fit_transform(X)
        X_t = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        
        total_output = y.shape[1]
        fixed_size = total_output // 4
        input_dim = X.shape[1]
        
        for i, s in enumerate(self.stokes_labels):
            print(f"\n=== Training Specialist for Stokes {s} ===")
            
            y_subset = y[:, i*fixed_size : (i+1)*fixed_size]
            
            mean_y = np.mean(y_subset)
            std_y = np.std(y_subset)
            min_std_threshold = 0.001 
            
            if std_y < min_std_threshold:
                scale_factor = 1.0 
            else:
                scale_factor = std_y

            self.scalers_y[s] = {'mean': mean_y, 'scale': scale_factor}
            
            y_norm = (y_subset - mean_y) / scale_factor
            y_t = torch.tensor(y_norm, dtype=torch.float32).to(self.device)
            
            full_dataset = TensorDataset(X_t, y_t)
            train_size = int(0.9 * len(full_dataset))
            val_size = len(full_dataset) - train_size
            
            generator = torch.Generator().manual_seed(self.random_state)
            train_ds, val_ds = random_split(full_dataset, [train_size, val_size], generator=generator)
            
            train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=self.batch_size*2, shuffle=False) 

            model = StokesResNet(
                input_dim=input_dim, 
                output_dim=fixed_size, 
                hidden_dim=self.hidden_dim, 
                n_blocks=self.n_blocks
            ).to(self.device)
            
            optimizer = optim.AdamW(model.parameters(), lr=self.lr, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.lr, total_steps=self.epochs*len(train_loader))
            
            criterion = PhysicsLoss(derivative_weight=self.deriv_weight)
            
            best_val_loss = float('inf')
            patience_counter = 0
            best_model_weights = None
            
            pbar = tqdm(range(self.epochs), desc=f"Training {s}", unit="epoch")
            
            for epoch in pbar:
                model.train()
                train_loss = 0.0
                
                for bx, by in train_loader:
                    optimizer.zero_grad()
                    pred = model(bx)
                    loss = criterion(pred, by)
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    train_loss += loss.item()
                
                train_loss /= len(train_loader)
                
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for bx, by in val_loader:
                        pred = model(bx)
                        loss = criterion(pred, by)
                        val_loss += loss.item()
                
                val_loss /= len(val_loader)
                
                pbar.set_postfix({'T_loss': f"{train_loss:.5f}", 'V_loss': f"{val_loss:.5f}"})
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_weights = copy.deepcopy(model.state_dict())
                    patience_counter = 0 
                else:
                    patience_counter += 1
                    
                if patience_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch+1}. Best Val Loss: {best_val_loss:.6f}")
                    break
            
            if best_model_weights is not None:
                model.load_state_dict(best_model_weights)
            
            self.models[s] = model
            
        return self
    
    def predict(self, X):
        # Conversione per evitare warning
        if hasattr(X, "values"): X = X.values 

        X_scaled = self.scaler_x.transform(X)
        X_t = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        
        dataset = TensorDataset(X_t)
        loader = DataLoader(dataset, batch_size=4096, shuffle=False)
        
        preds_dict = {s: [] for s in self.stokes_labels}
        
        with torch.no_grad():
            for s in self.stokes_labels:
                self.models[s].eval()
            
            for (bx,) in loader:
                for s in self.stokes_labels:
                    out_norm = self.models[s](bx).cpu().numpy()
                    
                    stats = self.scalers_y[s]
                    out_real = out_norm * stats['scale'] + stats['mean']
                    
                    preds_dict[s].append(out_real)
        
        full_preds = {}
        for s in self.stokes_labels:
            full_preds[s] = np.concatenate(preds_dict[s], axis=0)
            
        return np.hstack([full_preds[s] for s in self.stokes_labels])

class PcaXgbWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, pca_transformer, xgb_params):
        self.pca_transformer = pca_transformer
        self.xgb_params = xgb_params
        self.model = MultiOutputRegressor(xgb.XGBRegressor(**xgb_params), n_jobs=-1)

    def fit(self, X, y):
        print("Fitting Standard PCA on target profiles...")
        y_coeffs = self.pca_transformer.fit_transform(y)
        
        print(f"Training XGBoost on {y_coeffs.shape[1]} PCA coefficients...")
        self.model.fit(X, y_coeffs)
        
        return self

    def predict(self, X):
        y_coeffs_pred = self.model.predict(X)
        
        return self.pca_transformer.inverse_transform(y_coeffs_pred)