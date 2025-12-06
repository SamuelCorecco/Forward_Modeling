import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def split_stokes_profiles(y, fixed_size):
    return {
        'I': y[:, 0:fixed_size],
        'Q': y[:, fixed_size:2*fixed_size],
        'U': y[:, 2*fixed_size:3*fixed_size],
        'V': y[:, 3*fixed_size:4*fixed_size]
    }

def renormalize_batch(y, fixed_size, cont_left=2, cont_right=2):

    I = y[:, 0:fixed_size]
    mean_left = np.mean(I[:, :cont_left], axis=1)
    mean_right = np.mean(I[:, -fixed_size:][:, -cont_right:], axis=1) 
    mean_right = np.mean(I[:, -cont_right:], axis=1)
    cont_val = 0.5 * (mean_left + mean_right)
    cont_val[cont_val == 0] = 1.0
    
    cont_val = cont_val[:, np.newaxis]

    y_norm = y / cont_val
    
    return y_norm


def load_and_prepare_datasets(config):

    print(f"Loading FULL data from {config.DATASET_SAVE_PATH}...")
    data = np.load(config.DATASET_SAVE_PATH)
    X = data['parameters']
    y = data['profiles']

    X_train_np, X_test_np, y_train_full, y_test_full = train_test_split(
        X, y,
        test_size=0.2, 
        random_state=config.SEED
    )

    y_train_norm = renormalize_batch(y_train_full, config.FIXED_SIZE)
    y_test_norm  = renormalize_batch(y_test_full,  config.FIXED_SIZE)

    n_features = X.shape[1]
    col_names = [f"feat_{i}" for i in range(n_features)]
    X_train = pd.DataFrame(X_train_np, columns=col_names)
    X_test  = pd.DataFrame(X_test_np,  columns=col_names)

    y_train_dict = split_stokes_profiles(y_train_norm, config.FIXED_SIZE)
    y_test_dict  = split_stokes_profiles(y_test_norm,  config.FIXED_SIZE)

    return X_train, X_test, y_train_dict, y_test_dict