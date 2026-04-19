"""
analyze_model.py  —  Inspect a saved model checkpoint.

After training, the server writes:
  • final_model.pkl        — always the latest round
  • model_round_1.pkl, model_round_2.pkl, …  — per-round checkpoints

Usage (run from repo root):
    python baseline_model/analyze_model.py                   # inspect final_model.pkl
    python baseline_model/analyze_model.py model_round_3.pkl # inspect a specific round
"""

import sys
import os
import numpy as np
import cloudpickle

# Feature order matches FEATURE_COLUMNS in the hospital dataset_loader.py
# (40 raw features + 2 engineered features appended by medical_feature_engineering)
FEATURE_NAMES = [
    # --- Vital Signs ---
    "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2",
    # --- Blood Gases & Acid-Base ---
    "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2",
    # --- Electrolytes & Metabolic ---
    "AST", "BUN", "Alkalinephos", "Calcium", "Chloride", "Creatinine",
    "Bilirubin_direct", "Glucose", "Lactate", "Magnesium", "Phosphate",
    "Potassium", "Bilirubin_total", "TroponinI",
    # --- Hematology ---
    "Hct", "Hgb", "PTT", "WBC", "Fibrinogen", "Platelets",
    # --- Demographics & Context ---
    "Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS",
    # --- Engineered Features (appended by medical_feature_engineering) ---
    "ShockIndex",       # HR / (SBP + 1e-6)  — high value → hemodynamic instability
    "BUN_Creat_Ratio",  # BUN / (Creatinine + 1e-6) — elevated → kidney dysfunction
]


def _sep(char="-", width=62):
    print(char * width)


def analyze_model(model_path: str = "final_model.pkl") -> None:
    if not os.path.exists(model_path):
        print(f"ERROR: '{model_path}' not found.")
        print("Run training first, or pass a valid checkpoint path.")
        sys.exit(1)

    with open(model_path, "rb") as f:
        model = cloudpickle.load(f)

    print()
    _sep("=")
    print(f"  Model Analysis  —  {model_path}")
    _sep("=")

    # ── 1. Pipeline architecture ──────────────────────────────────────────────
    print("\n[1] Pipeline Steps")
    _sep()
    for step_name, step_obj in model.steps:
        print(f"  {step_name:<16}  {step_obj}")
    print()

    # ── 2. Scaler statistics ──────────────────────────────────────────────────
    scaler = model.named_steps["scaler"]
    clf = model.named_steps["clf"]

    n_scaled = len(scaler.mean_)
    names = FEATURE_NAMES[:n_scaled]

    # Warn if PolynomialFeatures was added (feature count explodes)
    if n_scaled > len(FEATURE_NAMES):
        print(
            f"  NOTE: {n_scaled} features detected (PolynomialFeatures likely active).\n"
            f"  Only the first {len(FEATURE_NAMES)} feature names are labelled.\n"
        )

    print(f"[2] StandardScaler  ({n_scaled} features after feature engineering)")
    _sep()
    print(f"  {'Feature':<22}  {'Mean':>12}  {'Std Dev':>12}")
    _sep()
    for i, fname in enumerate(names):
        std = float(np.sqrt(scaler.var_[i]))
        print(f"  {fname:<22}  {scaler.mean_[i]:>12.4f}  {std:>12.4f}")
    print()

    # ── 3. Logistic Regression weights ───────────────────────────────────────
    coef = clf.coef_[0]
    intercept = float(clf.intercept_[0])
    odds_ratios = np.exp(coef)

    # Truncate names list to actual coefficient count
    coef_names = names[:len(coef)]

    print(f"[3] Logistic Regression Weights  ({len(coef)} coefficients)")
    _sep()
    print(
        f"  Intercept:  {intercept:+.6f}  "
        f"(odds ratio = {np.exp(intercept):.4f})\n"
    )

    # Sort descending by absolute value so strongest predictors appear first
    order = np.argsort(np.abs(coef))[::-1]

    print(
        f"  {'Rank':<5} {'Feature':<22} {'Coefficient':>14}"
        f" {'Odds Ratio':>12} {'Direction'}"
    )
    _sep()
    for rank, idx in enumerate(order, 1):
        if idx >= len(coef_names):
            fname = f"feature_{idx}"
        else:
            fname = coef_names[idx]
        direction = "↑ sepsis risk" if coef[idx] > 0 else "↓ sepsis risk"
        print(
            f"  {rank:<5} {fname:<22} {coef[idx]:>+14.6f}"
            f" {odds_ratios[idx]:>12.4f}  {direction}"
        )
    print()

    # ── 4. Quick summary ──────────────────────────────────────────────────────
    pos_idx = [i for i in order if coef[i] > 0][:5]
    neg_idx = [i for i in order if coef[i] < 0][:5]

    def _name(i):
        return coef_names[i] if i < len(coef_names) else f"feature_{i}"

    print("[4] Quick Summary")
    _sep()
    print("  Top 5 features that INCREASE predicted sepsis risk:")
    for i in pos_idx:
        print(f"    +  {_name(i):<22}  coef = {coef[i]:+.6f}  (OR = {odds_ratios[i]:.4f})")
    print()
    print("  Top 5 features that DECREASE predicted sepsis risk:")
    for i in neg_idx:
        print(f"    -  {_name(i):<22}  coef = {coef[i]:+.6f}  (OR = {odds_ratios[i]:.4f})")
    print()
    print(f"  Total trainable parameters: {len(coef)} coefs + 1 intercept = {len(coef)+1}")
    print(f"  Model loaded from: {os.path.abspath(model_path)}")
    print()


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "final_model.pkl"
    analyze_model(path)
