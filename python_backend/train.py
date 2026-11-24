"""Train the four required models (plus XGBoost as the primary one).

Run from the backend folder:
    python train.py --data ../data/creditrisk_x_enhanced.csv

Uses the `split` column that comes with the CSV:
    train       -> fit models
    validation  -> early stopping for XGBoost + picking the cost-optimal threshold
    test        -> only used for the final numbers
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from costs import DEFAULT_COSTS, best_threshold, evaluate_at
from features import (ENGINEERED_COLUMNS, FEATURE_COLUMNS, RAW_COLUMNS, TARGET,
                      add_engineered_features, clean_categories)

SEED = 42
ARTIFACT_DIR = Path(__file__).parent / "artifacts"


def load_data(csv_path):
    df = pd.read_csv(csv_path)
    df = clean_categories(df)

    # sanity check: our feature builder should reproduce the engineered columns
    rebuilt = add_engineered_features(df)
    for col in ENGINEERED_COLUMNS:
        if not np.allclose(rebuilt[col], df[col], rtol=1e-6, atol=1e-6):
            raise ValueError(f"feature builder disagrees with the CSV on {col}")

    parts = {}
    for name in ("train", "validation", "test"):
        rows = df["split"] == name
        parts[name] = (df.loc[rows, FEATURE_COLUMNS].astype(float), df.loc[rows, TARGET].to_numpy())
    return parts


def build_models():
    return {
        "logreg": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, C=0.5)
        ),
        # LinearSVC has no probabilities, so wrap it in a calibrator. The RBF SVC
        # is much slower on 21k rows and barely better on this data.
        "svm": make_pipeline(
            StandardScaler(),
            CalibratedClassifierCV(LinearSVC(C=0.1, max_iter=5000), cv=5),
        ),
        # kept shallow on purpose so the rules are readable
        "tree": DecisionTreeClassifier(
            max_depth=4, min_samples_leaf=100, random_state=SEED
        ),
        "forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=20, n_jobs=-1, random_state=SEED
        ),
        "xgboost": XGBClassifier(
            n_estimators=800,
            learning_rate=0.03,
            max_depth=4,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=2.0,
            eval_metric="aucpr",
            early_stopping_rounds=40,
            n_jobs=-1,
            random_state=SEED,
        ),
    }


def tree_to_rules(tree, feature_names):
    """Turn a fitted decision tree into plain if-then rules, one per leaf."""
    t = tree.tree_
    rules = []

    def walk(node, bounds):
        if t.children_left[node] == -1:          # leaf
            counts = t.value[node][0]
            conditions = []
            for feat, (lo, hi) in bounds.items():
                if lo is not None and hi is not None:
                    conditions.append(f"{lo:.2f} < {feat} <= {hi:.2f}")
                elif lo is not None:
                    conditions.append(f"{feat} > {lo:.2f}")
                else:
                    conditions.append(f"{feat} <= {hi:.2f}")
            rules.append({
                "conditions": conditions,
                "default_rate": round(float(counts[1] / counts.sum()), 4),
                "customers": int(t.n_node_samples[node]),
            })
            return
        feat = feature_names[t.feature[node]]
        cut = float(t.threshold[node])
        lo, hi = bounds.get(feat, (None, None))
        walk(t.children_left[node], {**bounds, feat: (lo, cut if hi is None else min(hi, cut))})
        walk(t.children_right[node], {**bounds, feat: (cut if lo is None else max(lo, cut), hi)})

    walk(0, {})
    return sorted(rules, key=lambda r: -r["default_rate"])


def main(csv_path):
    parts = load_data(csv_path)
    X_train, y_train = parts["train"]
    X_val, y_val = parts["validation"]
    X_test, y_test = parts["test"]
    print(f"train={len(X_train)}  val={len(X_val)}  test={len(X_test)}  "
          f"default rate={y_train.mean():.3f}")

    models = build_models()
    val_proba, test_proba = {}, {}

    for name, model in models.items():
        print(f"fitting {name} ...")
        if name == "xgboost":
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            print(f"  best iteration: {model.best_iteration}")
        else:
            model.fit(X_train, y_train)
        val_proba[name] = model.predict_proba(X_val)[:, 1]
        test_proba[name] = model.predict_proba(X_test)[:, 1]

    # quick console summary (the dashboard recomputes this from the saved scores)
    print(f"\n{'model':10s} {'ROC-AUC':>8s} {'PR-AUC':>8s} {'thr':>5s} "
          f"{'test cost':>10s} {'cost@0.5':>10s}")
    for name in models:
        thr, _ = best_threshold(y_val, val_proba[name], DEFAULT_COSTS)
        at_best = evaluate_at(y_test, test_proba[name], thr, DEFAULT_COSTS)
        at_half = evaluate_at(y_test, test_proba[name], 0.5, DEFAULT_COSTS)
        print(f"{name:10s} {roc_auc_score(y_test, test_proba[name]):8.3f} "
              f"{average_precision_score(y_test, test_proba[name]):8.3f} "
              f"{thr:5.2f} {at_best['total_cost']:10.0f} {at_half['total_cost']:10.0f}")

    # --- things the dashboard needs for the "transparent rules" panel ---------
    tree_rules = tree_to_rules(models["tree"], FEATURE_COLUMNS)

    logreg = models["logreg"]
    coefs = logreg[-1].coef_[0]           # coefficients on standardized features
    order = np.argsort(-np.abs(coefs))[:12]
    logreg_top = [{"feature": FEATURE_COLUMNS[i], "coef": round(float(coefs[i]), 3)}
                  for i in order]

    xgb_importance = sorted(
        zip(FEATURE_COLUMNS, models["xgboost"].feature_importances_.tolist()),
        key=lambda pair: -pair[1],
    )[:12]

    # a few real test customers so the dashboard has something to pre-fill
    test_rows = X_test.copy()
    test_rows["proba"] = test_proba["xgboost"]
    test_rows["actual"] = y_test
    samples = {
        "low": test_rows.nsmallest(1, "proba")[RAW_COLUMNS].iloc[0].to_dict(),
        "medium": test_rows.iloc[(test_rows["proba"] - 0.35).abs().argsort()[:1]][RAW_COLUMNS].iloc[0].to_dict(),
        "high": test_rows.nlargest(1, "proba")[RAW_COLUMNS].iloc[0].to_dict(),
    }

    ARTIFACT_DIR.mkdir(exist_ok=True)
    joblib.dump(models, ARTIFACT_DIR / "models.joblib")
    np.savez(
        ARTIFACT_DIR / "scores.npz",
        y_val=y_val,
        y_test=y_test,
        **{f"val_{k}": v for k, v in val_proba.items()},
        **{f"test_{k}": v for k, v in test_proba.items()},
    )
    meta = {
        "features": FEATURE_COLUMNS,
        "tree_rules": tree_rules,
        "logreg_top": logreg_top,
        "xgb_importance": [{"feature": f, "importance": round(v, 4)} for f, v in xgb_importance],
        "samples": {k: {c: float(v) for c, v in row.items()} for k, row in samples.items()},
        "default_costs": DEFAULT_COSTS,
    }
    (ARTIFACT_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nsaved artifacts to {ARTIFACT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="../data/creditrisk_x_enhanced.csv")
    main(parser.parse_args().data)
