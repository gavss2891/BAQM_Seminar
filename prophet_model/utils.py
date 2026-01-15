import numpy as np

def calculate_smape(y_true, y_pred):
    """
    Calculates Symmetric Mean Absolute Percentage Error (sMAPE).
    Range: 0% to 200%
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    numerator = np.abs(y_pred - y_true) * 100
    denominator = np.abs(y_pred) + np.abs(y_true)
    
    # Safe division
    ratio = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0)
    
    return 2 * np.mean(ratio)