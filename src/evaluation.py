import numpy as np
from sklearn.metrics import mean_squared_error
import numpy as np

def concordance_correlation_coefficient(y_true, y_pred):
    """
    Lin's CCC: Penalizza differenze di scala e shift.
    Valore tra -1 e 1. 1 è fit perfetto (y=x).
    """
    # Calcolo su asse 1 (lunghezza profilo)
    mu_true = np.mean(y_true, axis=1)
    mu_pred = np.mean(y_pred, axis=1)
    
    var_true = np.var(y_true, axis=1)
    var_pred = np.var(y_pred, axis=1)
    
    sd_true = np.sqrt(var_true)
    sd_pred = np.sqrt(var_pred)
    
    # Covarianza
    covariance = np.mean((y_true - mu_true[:, None]) * (y_pred - mu_pred[:, None]), axis=1)
    
    # Pearson Correlation
    rho = covariance / (sd_true * sd_pred + 1e-4)
    
    # CCC Formula
    numerator = 2 * rho * sd_true * sd_pred
    denominator = var_true + var_pred + (mu_true - mu_pred)**2
    
    ccc = numerator / (denominator + 1e-4)
    return np.nanmean(ccc)


def peak_amplitude_error(y_true, y_pred):
    max_true = np.max(np.abs(y_true), axis=1)
    max_pred = np.max(np.abs(y_pred), axis=1)
    
    mask = max_true > 1e-2
    
    err = np.abs(max_pred[mask] - max_true[mask]) / max_true[mask]
    return np.mean(err) * 100 


def rmse_score(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def vec_pearson_corr(A, B):
    A_c = A - np.mean(A, axis=1, keepdims=True)
    B_c = B - np.mean(B, axis=1, keepdims=True)
    
    num = np.sum(A_c * B_c, axis=1)
    
    den = np.sqrt(np.sum(A_c**2, axis=1)) * np.sqrt(np.sum(B_c**2, axis=1))
    
    corr = np.divide(num, den, out=np.zeros_like(num), where=den!=0)
    return corr

def derivative_correlation(y_true, y_pred, stokes_type=None, threshold=0.0035):

    diff_true = np.diff(y_true, axis=1)
    diff_pred = np.diff(y_pred, axis=1)

    if stokes_type in ["Q", "U", "V"]:
        diff_true = np.where(np.abs(diff_true) < threshold, 0.0, diff_true)
        diff_pred = np.where(np.abs(diff_pred) < threshold, 0.0, diff_pred)

    corrs = vec_pearson_corr(diff_true, diff_pred)
    
    return np.nanmean(corrs)