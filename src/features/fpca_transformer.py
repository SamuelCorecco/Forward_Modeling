import numpy as np
import skfda
from skfda.preprocessing.dim_reduction.feature_extraction import FPCA
from sklearn.base import BaseEstimator, TransformerMixin

class SolarFpcaTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, fixed_size, var_ratio=0.99):
        self.fixed_size = fixed_size
        self.var_ratio = var_ratio
        
        self.fpca_models = {} 
        self.n_components_ = {}
        self.stokes = ['I', 'Q', 'U', 'V']

    def _split(self, y):
        return {
            'I': y[:, 0:self.fixed_size],
            'Q': y[:, self.fixed_size:2*self.fixed_size],
            'U': y[:, 2*self.fixed_size:3*self.fixed_size],
            'V': y[:, 3*self.fixed_size:4*self.fixed_size]
        }

    def _fit_single(self, matrix):
        grid_points = np.arange(matrix.shape[1])
        fd = skfda.FDataGrid(matrix, grid_points=grid_points)
        
        tmp_fpca = FPCA(n_components=min(matrix.shape[1], 50))
        tmp_fpca.fit(fd)
        cumvar = np.cumsum(tmp_fpca.explained_variance_ratio_)
        
        n_comp = np.searchsorted(cumvar, self.var_ratio) + 1
        
        final_fpca = FPCA(n_components=n_comp)
        final_fpca.fit(fd)
        return final_fpca

    def fit(self, y, X=None):
        data_dict = self._split(y)
        for s in self.stokes:
            self.fpca_models[s] = self._fit_single(data_dict[s])
            self.n_components_[s] = self.fpca_models[s].n_components
            print(f"[{s}] fPCA components kept: {self.n_components_[s]}")
        return self

    def transform(self, y):
        data_dict = self._split(y)
        coeffs_list = []
        for s in self.stokes:
            fd = skfda.FDataGrid(data_dict[s], grid_points=np.arange(self.fixed_size))
            coeffs = self.fpca_models[s].transform(fd)
            coeffs_list.append(coeffs)
        
        return np.hstack(coeffs_list)

    def inverse_transform(self, y_reduced):
        current_idx = 0
        rec_list = []
        
        for s in self.stokes:
            n_c = self.n_components_[s]
            coeffs_s = y_reduced[:, current_idx : current_idx + n_c]
            current_idx += n_c
            
            fd_rec = self.fpca_models[s].inverse_transform(coeffs_s)
            rec_matrix = fd_rec.data_matrix.squeeze()
            
            if rec_matrix.ndim == 1:
                rec_matrix = rec_matrix.reshape(1, -1)
                
            rec_list.append(rec_matrix)
            
        return np.hstack(rec_list)