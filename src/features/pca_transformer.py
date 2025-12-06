# src/features/pca_transformer.py
import numpy as np
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, TransformerMixin

class SolarPcaTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, fixed_size, n_components=0.99):
        self.fixed_size = fixed_size
        self.n_components = n_components
        self.pca_models = {}
        self.stokes = ['I', 'Q', 'U', 'V']
        self.n_components_kept = {} 

    def _split(self, y):
        return {
            'I': y[:, 0:self.fixed_size],
            'Q': y[:, self.fixed_size:2*self.fixed_size],
            'U': y[:, 2*self.fixed_size:3*self.fixed_size],
            'V': y[:, 3*self.fixed_size:4*self.fixed_size]
        }

    def fit(self, y, X=None):
        data_dict = self._split(y)
        for s in self.stokes:
            # Fit PCA
            self.pca_models[s] = PCA(n_components=self.n_components)
            self.pca_models[s].fit(data_dict[s])
            self.n_components_kept[s] = self.pca_models[s].n_components_
        return self

    def transform(self, y):
        data_dict = self._split(y)
        coeffs_list = []
        for s in self.stokes:
            coeffs = self.pca_models[s].transform(data_dict[s])
            coeffs_list.append(coeffs)
        return np.hstack(coeffs_list)

    def inverse_transform(self, y_coeffs):
        current_idx = 0
        rec_list = []
        for s in self.stokes:
            n_c = self.n_components_kept[s]
            coeffs_s = y_coeffs[:, current_idx : current_idx + n_c]
            current_idx += n_c
            
            rec = self.pca_models[s].inverse_transform(coeffs_s)
            rec_list.append(rec)
        return np.hstack(rec_list)