# -*- coding: utf-8 -*-
"""
Created on 2026 by EDI, ZHAW and UNAM

@author: Edi
"""

# -*- coding: utf-8 -*-
"""
Optimized Grid Search using NumPy Vectorization
"""

from itertools import product
import numpy as np
import pandas as pd

# ==========================================================
# LOAD DATA
# ==========================================================

df = pd.read_csv("events.csv")
y_true = df["Class"].values

metrics = ["A", "B", "C", "D"]

# Pre-extract numpy arrays for blazing-fast vectorized comparisons
vals_A = df["A"].values
vals_B = df["B"].values
vals_C = df["C"].values
vals_D = df["D"].values

# Pre-calculate masks for performance inside the loop
pos_mask = y_true == 1
neg_mask = y_true == 0

n_pos = np.sum(pos_mask)
n_neg = np.sum(neg_mask)

# ==========================================================
# GENERATE THRESHOLD CANDIDATES
# ==========================================================


def generate_thresholds(values):
    """Create midpoint thresholds between consecutive sorted unique observations."""
    values = np.sort(np.unique(values))

    if len(values) < 2:
        return values

    return (values[:-1] + values[1:]) / 2


candidate_thresholds = {
    metric: generate_thresholds(df[metric].values) for metric in metrics
}

# ==========================================================
# EXHAUSTIVE GRID SEARCH (WITH TIE-BREAKER)
# ==========================================================

best_score = -1
best_secondary_score = -1
best_result = None

# Using product over candidate arrays
for TA, TB, TC, TD in product(
    candidate_thresholds["A"],
    candidate_thresholds["B"],
    candidate_thresholds["C"],
    candidate_thresholds["D"],
):

    # Pre-compute individual boolean masks for the tie-breaker
    pred_A = vals_A > TA
    pred_B = vals_B > TB
    pred_C = vals_C > TC
    pred_D = vals_D > TD

    # Entirely vectorized vote calculation across all rows simultaneously
    votes = (
        pred_A.astype(np.int8)
        + pred_B.astype(np.int8)
        + pred_C.astype(np.int8)
        + pred_D.astype(np.int8)
    )

    y_pred = votes >= 2

    # Fast NumPy counting using pre-computed masks
    tp = np.sum(y_pred[pos_mask])
    fp = np.sum(y_pred[neg_mask])
    fn = n_pos - tp
    tn = n_neg - fp

    # Balanced accuracy calculation (Primary Objective)
    sensitivity = tp / n_pos if n_pos > 0 else 0
    specificity = tn / n_neg if n_neg > 0 else 0
    ba = 0.5 * (sensitivity + specificity)

    # Secondary Objective: Maximize individual feature accuracy
    # Counts how many correct decisions all features make individually.
    indiv_tp_sum = (
        np.sum(pred_A[pos_mask])
        + np.sum(pred_B[pos_mask])
        + np.sum(pred_C[pos_mask])
        + np.sum(pred_D[pos_mask])
    )
    indiv_tn_sum = (
        np.sum(~pred_A[neg_mask])
        + np.sum(~pred_B[neg_mask])
        + np.sum(~pred_C[neg_mask])
        + np.sum(~pred_D[neg_mask])
    )
    secondary_score = indiv_tp_sum + indiv_tn_sum

    # Update logic: If BA is better, OR if BA is identical but individual alignment is superior
    if (ba > best_score) or (
        np.isclose(ba, best_score) and secondary_score > best_secondary_score
    ):
        best_score = ba
        best_secondary_score = secondary_score
        best_result = {
            "thresholds": {"A": TA, "B": TB, "C": TC, "D": TD},
            "BA": ba,
            "TP": tp,
            "TN": tn,
            "FP": fp,
            "FN": fn,
        }

# ==========================================================
# RESULTS
# ==========================================================

print("\n==============================")
print("OPTIMAL THRESHOLDS")
print("==============================")

for metric, value in best_result["thresholds"].items():
    print(f"{metric}: {value:.6f}")

print("\nBalanced Accuracy:")
print(f"{best_result['BA']:.6f}")

print("\nConfusion Matrix")
print(f"TP = {best_result['TP']}")
print(f"TN = {best_result['TN']}")
print(f"FP = {best_result['FP']}")
print(f"FN = {best_result['FN']}")

print("\nSensitivity:")
print(f"{best_result['TP'] / (best_result['TP'] + best_result['FN']):.6f}")

print("\nSpecificity:")
print(f"{best_result['TN'] / (best_result['TN'] + best_result['FP']):.6f}")