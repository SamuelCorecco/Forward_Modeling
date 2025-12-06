import numpy as np
import torch

from src.features.fpca_transformer import SolarFpcaTransformer
from src.features.pca_transformer import SolarPcaTransformer
from src.models.wrappers import FpcaXgbWrapper, Paper2008MlpWrapper, StandardXgbWrapper, MultiResNetWrapper, PcaXgbWrapper

    
def build_model(config_dict):

    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
        
    model_type = config_dict.get('model_type', 'xgboost') # 
    
    if model_type == 'xgboost':
        fpca_params = config_dict.get('fpca', {})
        transformer = SolarFpcaTransformer(
            fixed_size=config_dict['dataset']['fixed_size'],
            var_ratio=fpca_params.get('var_ratio', 0.99)
        )
        
        xgb_params = config_dict.get('xgboost', {}).copy()
        xgb_params['random_state'] = config_dict['dataset']['seed']
        
        return FpcaXgbWrapper(transformer, xgb_params)

    elif model_type == 'paper_mlp':
        pca_params = config_dict.get('pca', {})
        transformer = SolarPcaTransformer(
            fixed_size=config_dict['dataset']['fixed_size'],
            n_components=pca_params.get('n_components', 0.99)
        )
        
        mlp_conf = config_dict.get('mlp', {}).copy()
        
        arch_iv = tuple(mlp_conf.pop('architecture_IV', [50, 35]))
        arch_qu = tuple(mlp_conf.pop('architecture_QU', [55, 55]))
        
        return Paper2008MlpWrapper(
            pca_transformer=transformer,
            mlp_params=mlp_conf,
            arch_iv=arch_iv,
            arch_qu=arch_qu
        )

    elif model_type == 'xgboost_full':
        xgb_params = config_dict.get('xgboost', {}).copy()
        xgb_params['random_state'] = config_dict['dataset']['seed']
        
        return StandardXgbWrapper(xgb_params)
    
    elif model_type == 'resnet':
        params = config_dict.get('resnet', {}) 
        
        return MultiResNetWrapper(
            hidden_dim=params.get('hidden_dim', 256),
            n_blocks=params.get('n_blocks', 4),
            lr=params.get('learning_rate', 1e-3),
            batch_size=params.get('batch_size', 256),
            epochs=params.get('epochs', 100),
            deriv_weight=params.get('derivative_weight', 1.0),
            patience=params.get('patience', 10),
            random_state=config_dict['dataset']['seed'],
            device='cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
        )

    elif model_type == 'pca_xgboost':
        pca_params = config_dict.get('pca', {})
        transformer = SolarPcaTransformer(
            fixed_size=config_dict['dataset']['fixed_size'],
            n_components=pca_params.get('n_components', 0.99)
        )
        
        xgb_params = config_dict.get('xgboost', {}).copy()
        xgb_params['random_state'] = config_dict['dataset']['seed']
        
        return PcaXgbWrapper(transformer, xgb_params)
    
    else:
        raise ValueError(f"Modello '{model_type}' non riconosciuto nel builder.")