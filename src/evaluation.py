from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.calibration import calibration_curve

def outcome_label(row: pd.Series) -> int:
    # 0=home, 1=draw, 2=away
    if row["home_goals"] > row["away_goals"]:
        return 0
    if row["home_goals"] == row["away_goals"]:
        return 1
    return 2

def add_true_outcome(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["y_true"] = df.apply(outcome_label, axis=1)
    return df

def multiclass_brier(probs: np.ndarray, y_true: np.ndarray, n_classes: int = 3) -> float:
    y_onehot = np.zeros((len(y_true), n_classes))
    y_onehot[np.arange(len(y_true)), y_true] = 1.0
    return np.mean(np.sum((probs - y_onehot) ** 2, axis=1))

def expected_value(prob: float, odds: float) -> float:
    return prob * odds - 1.0

def best_ev_bet(model_probs: np.ndarray, odds_triplet: np.ndarray) -> tuple[int, float]:
    evs = model_probs * odds_triplet - 1.0
    i = int(np.argmax(evs))
    return i, float(evs[i])

def calibration_data_binary(p: np.ndarray, y: np.ndarray, n_bins: int = 10):
    frac_pos, mean_pred = calibration_curve(y, p, n_bins=n_bins, strategy="uniform")
    return mean_pred, frac_pos

def summarize_probs(probs: np.ndarray, y_true: np.ndarray) -> dict:
    return {
        "log_loss": float(log_loss(y_true, probs, labels=[0, 1, 2])),
        "brier": float(multiclass_brier(probs, y_true)),
    }
