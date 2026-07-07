"""
Compares an incoming data batch's feature distributions against the
reference (training) distribution using a two-sample Kolmogorov-Smirnov
test per feature. Flags any feature where p < 0.05 as "drifted".

Runs two scenarios to prove the check works both ways:
1. new_batch.csv  -> genuinely held-out data, same distribution as
                      reference, expect ~no drift.
2. drifted_batch  -> new_batch with a synthetic shift applied to
                      alcohol and volatile acidity, expect drift
                      flagged on exactly those two columns.
"""
import json
import pandas as pd
from scipy.stats import ks_2samp

FEATURE_COLUMNS = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide",
    "density", "pH", "sulphates", "alcohol",
]
ALPHA = 0.05


def check_drift(reference, batch):
    results = {}
    drifted_features = []

    for col in FEATURE_COLUMNS:
        stat, p_value = ks_2samp(reference[col].dropna(), batch[col].dropna())
        drifted = bool(p_value < ALPHA)
        results[col] = {
            "ks_statistic": round(float(stat), 4),
            "p_value": round(float(p_value), 4),
            "drifted": drifted,
        }
        if drifted:
            drifted_features.append(col)

    return {
        "n_drifted": len(drifted_features),
        "drifted_features": drifted_features,
        "details": results,
    }


def print_report(label, report):
    print(f"\n=== {label} ===")
    if report["n_drifted"] == 0:
        print("No significant drift detected.")
    else:
        print(f"Drift detected in {report['n_drifted']} feature(s): {report['drifted_features']}")
    for col, res in report["details"].items():
        flag = "DRIFT" if res["drifted"] else "ok"
        print(f"  [{flag}] {col}: KS={res['ks_statistic']}, p={res['p_value']}")


def main():
    reference = pd.read_csv("data/reference.csv")
    new_batch = pd.read_csv("data/new_batch.csv")

    # Synthetic drift: simulate a process shift, e.g. a new supplier
    # or a measurement calibration change.
    drifted_batch = new_batch.copy()
    drifted_batch["alcohol"] = drifted_batch["alcohol"] + 1.5
    drifted_batch["volatile acidity"] = drifted_batch["volatile acidity"] + 0.4

    report_clean = check_drift(reference, new_batch)
    report_drifted = check_drift(reference, drifted_batch)

    print_report("new_batch vs reference (expect no drift)", report_clean)
    print_report("drifted_batch vs reference (expect drift on alcohol, volatile acidity)", report_drifted)

    with open("reports/drift_report.json", "w") as f:
        json.dump({"new_batch": report_clean, "drifted_batch": report_drifted}, f, indent=2)
    print("\nSaved combined report to reports/drift_report.json")


if __name__ == "__main__":
    main()