"""
random_forest_analysis.py
---------------------------
A single, focused script that tells the complete Random Forest story:
  1. Load & prepare the real Credit Card Fraud dataset
  2. Train Random Forest with balanced class weights
  3. Evaluate: Accuracy, Precision, Recall, F1, ROC-AUC
  4. Save a single, publication-quality 6-panel visualization

RUN:
    python random_forest_analysis.py

OUTPUT:
    random_forest_results.png  — save this for your report/GitHub README
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve,
    average_precision_score
)

# ── make sure backend.utils is importable regardless of where you run from ──
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backend.utils.preprocessing import (
    load_raw_dataset, handle_missing_values,
    engineer_features, get_feature_columns, TARGET_COLUMN
)

# ============================================================
# DESIGN TOKENS  (dark, data-dense, consistent with dashboard)
# ============================================================
BG       = "#0b0e14"
PANEL    = "#131826"
CARD     = "#1c2333"
BORDER   = "#2a3146"
TEXT     = "#e2e8f0"
MUTED    = "#94a3b8"
TEAL     = "#5eead4"
RED      = "#f43f5e"
AMBER    = "#fbbf24"
PURPLE   = "#a78bfa"

plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    PANEL,
    "axes.edgecolor":    BORDER,
    "axes.labelcolor":   MUTED,
    "axes.titlecolor":   TEXT,
    "xtick.color":       MUTED,
    "ytick.color":       MUTED,
    "text.color":        TEXT,
    "grid.color":        BORDER,
    "grid.linewidth":    0.6,
    "font.family":       "DejaVu Sans",
    "font.size":         10,
})

# ============================================================
# 1. LOAD & PREPARE DATA
# ============================================================
print("=" * 60)
print("  Random Forest — Fraud Detection Analysis")
print("=" * 60)

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "raw", "creditcard.csv")
print(f"\n[1/4] Loading dataset from {DATA_PATH} ...")

df = load_raw_dataset(DATA_PATH)
df = handle_missing_values(df)
df = engineer_features(df)

feature_cols = get_feature_columns(df)
X = df[feature_cols]
y = df[TARGET_COLUMN]

print(f"      Total transactions : {len(y):,}")
print(f"      Legitimate         : {(y==0).sum():,}  ({(y==0).mean()*100:.2f}%)")
print(f"      Fraudulent         : {(y==1).sum():,}  ({(y==1).mean()*100:.4f}%)")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

scaler = StandardScaler()
X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols)
X_test_s  = pd.DataFrame(scaler.transform(X_test),      columns=feature_cols)

# ============================================================
# 2. TRAIN RANDOM FOREST
# ============================================================
print("\n[2/4] Training Random Forest (class_weight='balanced') ...")
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    class_weight="balanced",   # key: compensates for 0.17% fraud rate
    random_state=42,
    n_jobs=-1,
)
rf.fit(X_train_s, y_train)
print("      Training complete.")

# ============================================================
# 3. EVALUATE
# ============================================================
print("\n[3/4] Evaluating on held-out test set ...")

y_pred  = rf.predict(X_test_s)
y_proba = rf.predict_proba(X_test_s)[:, 1]

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)
roc_auc   = roc_auc_score(y_test, y_proba)
avg_prec  = average_precision_score(y_test, y_proba)
cm        = confusion_matrix(y_test, y_pred)

print(f"\n  Accuracy  : {accuracy:.4f}")
print(f"  Precision : {precision:.4f}  (of flagged txns, this % are real fraud)")
print(f"  Recall    : {recall:.4f}  (of real fraud cases, this % are caught)")
print(f"  F1-Score  : {f1:.4f}")
print(f"  ROC-AUC   : {roc_auc:.4f}")
print(f"  Avg Prec  : {avg_prec:.4f}")

# Feature importances
feat_imp = pd.Series(rf.feature_importances_, index=feature_cols)
top10    = feat_imp.sort_values(ascending=False).head(10)

# ============================================================
# 4. VISUALIZE — 6-panel figure
# ============================================================
print("\n[4/4] Building visualisation ...")

fig = plt.figure(figsize=(18, 13), facecolor=BG)
fig.suptitle(
    "Random Forest — Credit Card Fraud Detection",
    fontsize=20, fontweight="bold", color=TEXT, y=0.98
)

gs = gridspec.GridSpec(
    3, 3,
    figure=fig,
    hspace=0.52,
    wspace=0.38,
    top=0.93, bottom=0.06, left=0.07, right=0.97
)

# ── helper: draw a subtle rounded card behind an axes ──────
def card_bg(ax):
    ax.set_facecolor(CARD)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
        spine.set_linewidth(0.8)

# ──────────────────────────────────────────────────────────
# PANEL A  — Metric scorecard (top-left, spans 1 col)
# ──────────────────────────────────────────────────────────
ax_scores = fig.add_subplot(gs[0, 0])
card_bg(ax_scores)
ax_scores.set_xlim(0, 1)
ax_scores.set_ylim(0, 1)
ax_scores.axis("off")
ax_scores.set_title("Model Metrics", fontsize=12, fontweight="bold",
                     color=TEXT, pad=10)

metrics = [
    ("Accuracy",  accuracy,  TEAL),
    ("Precision", precision, AMBER),
    ("Recall",    recall,    RED),
    ("F1-Score",  f1,        PURPLE),
    ("ROC-AUC",   roc_auc,   TEAL),
]
for i, (label, val, color) in enumerate(metrics):
    y_pos = 0.85 - i * 0.18
    # bar track
    ax_scores.add_patch(FancyBboxPatch(
        (0.05, y_pos - 0.04), 0.9, 0.10,
        boxstyle="round,pad=0.01",
        facecolor=PANEL, edgecolor=BORDER, linewidth=0.6
    ))
    # filled bar
    ax_scores.add_patch(FancyBboxPatch(
        (0.05, y_pos - 0.04), 0.9 * val, 0.10,
        boxstyle="round,pad=0.01",
        facecolor=color, alpha=0.25, edgecolor="none"
    ))
    ax_scores.text(0.07, y_pos + 0.01, label,
                   color=MUTED, fontsize=9, va="center")
    ax_scores.text(0.93, y_pos + 0.01, f"{val:.3f}",
                   color=color, fontsize=10, fontweight="bold",
                   va="center", ha="right")

# ──────────────────────────────────────────────────────────
# PANEL B  — Class distribution (top-middle)
# ──────────────────────────────────────────────────────────
ax_cls = fig.add_subplot(gs[0, 1])
card_bg(ax_cls)
counts = [int((y==0).sum()), int((y==1).sum())]
bars = ax_cls.bar(["Legitimate", "Fraud"], counts,
                   color=[TEAL, RED], alpha=0.85,
                   width=0.5, edgecolor=BORDER)
ax_cls.set_yscale("log")
ax_cls.set_title("Class Distribution (log scale)", fontsize=11,
                  fontweight="bold", color=TEXT)
ax_cls.set_ylabel("Count (log)", color=MUTED)
for bar, count in zip(bars, counts):
    ax_cls.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() * 1.3,
                f"{count:,}", ha="center", va="bottom",
                color=TEXT, fontsize=9, fontweight="bold")

# ──────────────────────────────────────────────────────────
# PANEL C  — Confusion matrix (top-right)
# ──────────────────────────────────────────────────────────
ax_cm = fig.add_subplot(gs[0, 2])
card_bg(ax_cm)
im = ax_cm.imshow(cm, cmap="Blues", aspect="auto")
ax_cm.set_xticks([0, 1]); ax_cm.set_xticklabels(["Pred: Legit", "Pred: Fraud"])
ax_cm.set_yticks([0, 1]); ax_cm.set_yticklabels(["Act: Legit", "Act: Fraud"])
ax_cm.set_title("Confusion Matrix", fontsize=11, fontweight="bold", color=TEXT)
labels = [["TN", "FP"], ["FN", "TP"]]
for i in range(2):
    for j in range(2):
        ax_cm.text(j, i, f"{labels[i][j]}\n{cm[i,j]:,}",
                   ha="center", va="center",
                   color=TEXT if cm[i,j] < cm.max()/2 else BG,
                   fontsize=10, fontweight="bold")

# ──────────────────────────────────────────────────────────
# PANEL D  — ROC curve (middle-left)
# ──────────────────────────────────────────────────────────
ax_roc = fig.add_subplot(gs[1, 0])
card_bg(ax_roc)
fpr, tpr, _ = roc_curve(y_test, y_proba)
ax_roc.plot(fpr, tpr, color=TEAL, lw=2,
            label=f"Random Forest (AUC = {roc_auc:.3f})")
ax_roc.plot([0,1],[0,1], "--", color=MUTED, lw=1, alpha=0.5, label="Random guess")
ax_roc.fill_between(fpr, tpr, alpha=0.08, color=TEAL)
ax_roc.set_xlabel("False Positive Rate")
ax_roc.set_ylabel("True Positive Rate")
ax_roc.set_title("ROC Curve", fontsize=11, fontweight="bold", color=TEXT)
ax_roc.legend(fontsize=8, facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT)
ax_roc.grid(True, alpha=0.3)

# ──────────────────────────────────────────────────────────
# PANEL E  — Precision-Recall curve (middle-center)
# ──────────────────────────────────────────────────────────
ax_pr = fig.add_subplot(gs[1, 1])
card_bg(ax_pr)
prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_proba)
ax_pr.plot(rec_curve, prec_curve, color=AMBER, lw=2,
           label=f"AP = {avg_prec:.3f}")
ax_pr.axhline(y_test.mean(), color=MUTED, lw=1, linestyle="--",
              label=f"Baseline ({y_test.mean():.4f})")
ax_pr.fill_between(rec_curve, prec_curve, alpha=0.08, color=AMBER)
ax_pr.set_xlabel("Recall")
ax_pr.set_ylabel("Precision")
ax_pr.set_title("Precision-Recall Curve", fontsize=11, fontweight="bold", color=TEXT)
ax_pr.legend(fontsize=8, facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT)
ax_pr.grid(True, alpha=0.3)

# ──────────────────────────────────────────────────────────
# PANEL F  — Fraud risk score distribution (middle-right)
# ──────────────────────────────────────────────────────────
ax_dist = fig.add_subplot(gs[1, 2])
card_bg(ax_dist)
bins = np.linspace(0, 1, 40)
ax_dist.hist(y_proba[y_test == 0], bins=bins, color=TEAL,
             alpha=0.6, label="Legitimate", density=True)
ax_dist.hist(y_proba[y_test == 1], bins=bins, color=RED,
             alpha=0.7, label="Fraud", density=True)
ax_dist.axvline(0.5, color=AMBER, lw=1.5, linestyle="--", label="Threshold 0.5")
ax_dist.set_xlabel("Predicted Fraud Probability")
ax_dist.set_ylabel("Density")
ax_dist.set_title("Risk Score Distribution", fontsize=11, fontweight="bold", color=TEXT)
ax_dist.legend(fontsize=8, facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT)
ax_dist.grid(True, alpha=0.3)

# ──────────────────────────────────────────────────────────
# PANEL G  — Top 10 feature importances (bottom, full width)
# ──────────────────────────────────────────────────────────
ax_fi = fig.add_subplot(gs[2, :])
card_bg(ax_fi)
colors = [RED if v > top10.mean() else TEAL for v in top10.values]
bars = ax_fi.barh(top10.index[::-1], top10.values[::-1],
                   color=colors[::-1], alpha=0.85,
                   edgecolor=BORDER, linewidth=0.5)
for bar, val in zip(bars, top10.values[::-1]):
    ax_fi.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
               f"{val:.4f}", va="center", color=MUTED, fontsize=8)
ax_fi.set_xlabel("Importance Score (mean decrease in impurity)")
ax_fi.set_title("Top 10 Most Important Features for Fraud Detection",
                fontsize=11, fontweight="bold", color=TEXT)
ax_fi.grid(True, axis="x", alpha=0.3)
ax_fi.set_xlim(0, top10.values.max() * 1.15)

# ── footer ──────────────────────────────────────────────
fig.text(0.5, 0.01,
         f"Dataset: Kaggle Credit Card Fraud  |  "
         f"Train: {len(X_train):,} txns  |  Test: {len(X_test):,} txns  |  "
         f"Fraud rate: {y.mean()*100:.4f}%  |  "
         f"Model: RandomForestClassifier(n_estimators=200, class_weight='balanced')",
         ha="center", fontsize=8, color=MUTED)

# ── save ────────────────────────────────────────────────
OUT = os.path.join(os.path.dirname(__file__), "random_forest_results.png")
plt.savefig(OUT, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()

print(f"\n  Saved → {OUT}")
print("\n" + "=" * 60)
print("  SUMMARY")
print("=" * 60)
print(f"  Accuracy  : {accuracy*100:.2f}%")
print(f"  Precision : {precision*100:.2f}%  — of every 100 fraud alerts, {precision*100:.0f} are real")
print(f"  Recall    : {recall*100:.2f}%  — catches {recall*100:.0f} out of every 100 fraud cases")
print(f"  F1-Score  : {f1:.4f}")
print(f"  ROC-AUC   : {roc_auc:.4f}")
print("=" * 60)
print(f"\n  Open random_forest_results.png to see the full visualisation.")
