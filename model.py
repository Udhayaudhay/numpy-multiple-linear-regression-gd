"""
NumPy Multiple Linear Regression GD

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - shuffle_xy
import numpy as np


def shuffle_xy(X, y, seed=42):
    """
    Randomly permute feature rows and targets together.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n, d)

    y : np.ndarray
        Target vector of shape (n,)

    seed : int
        Random seed for reproducibility

    Returns
    -------
    X_shuffled : np.ndarray
        Shuffled feature matrix

    y_shuffled : np.ndarray
        Shuffled target vector
    """

    # Create reproducible random number generator
    rng = np.random.default_rng(seed)

    # Generate one random permutation of row indices
    indices = rng.permutation(len(X))

    # Apply the SAME permutation to X and y
    X_shuffled = X[indices]
    y_shuffled = y[indices]

    return X_shuffled, y_shuffled

# Step 2 - split_train_val_test
def split_train_val_test(X, y, train_frac=0.6, val_frac=0.2):
    # TODO: Slice already-shuffled data into contiguous train/val/test partitions...
    n=len(X)
    n_train=int(n*train_frac)
    n_val=int(n*val_frac)
    X_train=X[:n_train]
    y_train=y[:n_train]

    X_val = X[n_train:n_train + n_val]
    y_val = y[n_train:n_train + n_val]

    X_test = X[n_train + n_val:]
    y_test = y[n_train + n_val:]

    return X_train, y_train, X_val, y_val, X_test, y_test

# Step 3 - compute_feature_stats
def compute_feature_stats(X):
    # TODO: Compute per-feature mean and std; replace std of 0 with 1
    mean=np.mean(X,axis=0)
    std=np.std(X,axis=0,ddof=0)
    std[std == 0] = 1.0
    return mean,std

# Step 4 - standardize_features
def standardize_features(X, mean, std):
    # TODO: Apply z-score normalization using precomputed training mean and std.
    X_scaled = (X - mean) / std
    return X_scaled

# Step 5 - add_bias_column
def add_bias_column(X):
    # TODO: Prepend a column of ones to feature matrix X
    n = X.shape[0]
    ones = np.ones((n, 1))
    return np.hstack((ones, X))

# Step 6 - prepare_design_matrix
def prepare_design_matrix(X, mean, std):
    # TODO: Standardize features then add the bias column to form the design matrix.
    X_scaled = (X - mean) / std
    return np.hstack((np.ones((X.shape[0], 1)), X_scaled))

# Step 7 - predict_linear
def predict_linear(X, weights):
    """Compute linear predictions y_hat = X @ weights.

    Args:
        X: Design matrix of shape (n, d_in), often including a bias column.
        weights: Weight vector of shape (d_in,).

    Returns:
        Predicted targets of shape (n,).
    """
    # TODO: Return the predicted target vector from X and weights
    return X @ weights

# Step 8 - mse_loss
def mse_loss(y_true, y_pred):
    # TODO: Return the average of squared residuals as a scalar float.
    residuals = y_true - y_pred
    squared_errors = residuals ** 2
    return float(np.mean(squared_errors))

# Step 9 - mse_gradient
def mse_gradient(X, y_true, y_pred):
    # TODO: Return the analytic MSE gradient w.r.t. weights: (2/n) X^T (y_pred - y_true)
    n=X.shape[0]
    residuals=y_pred-y_true
    gradeint=(2/n) *( X.T @ residuals)
    return gradeint

# Step 10 - normal_equation
def normal_equation(X, y):
    # TODO: Solve for the closed-form least-squares weights via the normal equation.
    A=X.T @ X
    B=X.T @ y
    w=np.linalg.solve(A,B)
    return w

# Step 11 - initialize_weights
def initialize_weights(n_features, seed=None):
    # TODO: Return (n_features,) weights sampled from N(0, 0.01)

    rng = np.random.default_rng(seed)
    return rng.normal(loc=0.0, scale=0.01, size=n_features)

# Step 12 - gd_step
def gd_step(X, y, weights, lr):
    """Run one full-batch gradient descent update on the weights.

    Args:
        X: Design matrix of shape (n, d_in).
        y: Target vector of shape (n,).
        weights: Current weight vector of shape (d_in,).
        lr: Learning rate (float).

    Returns:
        Updated weight vector of shape (d_in,).
    """
    # TODO: return the updated weight vector after one MSE gradient step
    y_pred = X @ weights
    n = X.shape[0]
    gradient = (2 / n) * (X.T @ (y_pred - y))
    updated_weights = weights - lr * gradient
    return updated_weights

# Step 13 - epoch_train_val_losses
def epoch_train_val_losses(X_train, y_train, X_val, y_val, weights):
    """Evaluate MSE on train and validation sets for the current weights.

    Args:
        X_train: Training design matrix of shape (n_tr, d_in).
        y_train: Training targets of shape (n_tr,).
        X_val: Validation design matrix of shape (n_va, d_in).
        y_val: Validation targets of shape (n_va,).
        weights: Weight vector of shape (d_in,).

    Returns:
        (train_loss, val_loss) as plain floats.
    """
    # TODO: return the pair (train_loss, val_loss) as MSE floats
    train_pred = X_train @ weights
    val_pred = X_val @ weights

    train_loss = np.mean((y_train - train_pred) ** 2)
    val_loss = np.mean((y_val - val_pred) ** 2)

    return float(train_loss), float(val_loss)

# Step 14 - update_early_stop_state (not yet solved)
# TODO: implement

# Step 15 - init_training_state (not yet solved)
# TODO: implement

# Step 16 - run_one_epoch (not yet solved)
# TODO: implement

# Step 17 - train_batch_gd (not yet solved)
# TODO: implement

# Step 18 - mean_absolute_error (not yet solved)
# TODO: implement

# Step 19 - root_mean_squared_error (not yet solved)
# TODO: implement

# Step 20 - r_squared (not yet solved)
# TODO: implement

# Step 21 - evaluate_regression (not yet solved)
# TODO: implement

# Step 22 - learning_curve_data (not yet solved)
# TODO: implement

# Step 23 - weights_l2_distance (not yet solved)
# TODO: implement

# Step 24 - create_lr_model (not yet solved)
# TODO: implement

# Step 25 - fit_lr_model (not yet solved)
# TODO: implement

# Step 26 - predict_lr_model (not yet solved)
# TODO: implement

# Step 27 - score_lr_model (not yet solved)
# TODO: implement

# Step 28 - compare_with_normal_equation (not yet solved)
# TODO: implement

