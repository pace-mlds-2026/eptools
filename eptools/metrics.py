import numpy as np
import pandas as pd


def wmape(y_true: pd.Series, y_pred: pd.Series) -> float:
    """returns weighted mean absolute percentage error
    """
    return float(np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)))
