# ============================================================
# STEP 3: FITNESS FUNCTION & CLASSIFIER
# Member 3's responsibility
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (f1_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)
import warnings
warnings.filterwarnings('ignore')


# ── Load Preprocessed Data ────────────────────────────────────
X = pd.read_csv('ga_project/X_scaled.csv')
y = pd.read_csv('ga_project/y.csv').squeeze()

feature_names = list(X.columns)
X_arr = X.values
y_arr = y.values

print("Data loaded:", X_arr.shape, "| Target:", y_arr.shape)


# ── Fitness Function ──────────────────────────────────────────
def fitness_function(chromosome):
    """
    Evaluate a binary chromosome by:
    1. Selecting features where chromosome[i] == 1
    2. Training a Random Forest with 5-fold CV
    3. Returning mean F1 score (macro average)

    Parameters
    ----------
    chromosome : np.ndarray of shape (13,)  — binary

    Returns
    -------
    float : mean F1 score (0.0 if no features selected)
    """
    selected_indices = np.where(chromosome == 1)[0]
    if len(selected_indices) == 0:
        return 0.0

    X_selected = X_arr[:, selected_indices]

    clf = RandomForestClassifier(
        n_estimators=10,  # Reduced from 50 to 10 for faster fitness evaluation
        max_depth=5,
        random_state=42,
        n_jobs=-1
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X_selected, y_arr,
                             cv=cv, scoring='f1_macro')
    return float(scores.mean())


# ── Baseline: All 13 Features ─────────────────────────────────
print("\n" + "=" * 55)
print("BASELINE: ALL 13 FEATURES")
print("=" * 55)

baseline_chrom = np.ones(13, dtype=int)
baseline_f1    = fitness_function(baseline_chrom)
print(f"Baseline F1 Score (all features): {baseline_f1:.4f}")


# ── Evaluate Best GA Chromosome ───────────────────────────────
def evaluate_chromosome(chromosome, label="GA Selected Features"):
    """Full evaluation with classification report and confusion matrix."""
    selected_indices = np.where(chromosome == 1)[0]
    selected_features = [feature_names[i] for i in selected_indices]

    print(f"\n{'='*55}")
    print(f"EVALUATION: {label}")
    print(f"{'='*55}")
    print(f"Selected features ({len(selected_features)}): {selected_features}")

    X_selected = X_arr[:, selected_indices]

    # Split manually for final report (80/20)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y_arr, test_size=0.2, random_state=42, stratify=y_arr
    )

    clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    f1  = f1_score(y_test, y_pred, average='macro')
    print(f"\nF1 Score (test set): {f1:.4f}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred, target_names=['No Disease','Disease'])}")

    return clf, y_test, y_pred, selected_features, f1


# ── Plot: Generation vs F1 ────────────────────────────────────
def plot_evolution(ga):
    """Plot how F1 improved across generations."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Genetic Algorithm — Evolution Results', fontsize=14, fontweight='bold')

    gens = range(1, len(ga.best_fitness_history) + 1)

    # Left: F1 over generations
    axes[0].plot(gens, ga.best_fitness_history, color='#1D9E75', linewidth=2, label='Best F1')
    axes[0].plot(gens, ga.avg_fitness_history,  color='#3B8BD4', linewidth=1.5,
                 linestyle='--', label='Avg F1', alpha=0.7)
    axes[0].axhline(y=baseline_f1, color='#E24B4A', linestyle=':', linewidth=1.5,
                    label=f'Baseline (all 13 features): {baseline_f1:.3f}')
    axes[0].set_xlabel('Generation')
    axes[0].set_ylabel('F1 Score')
    axes[0].set_title('F1 Score Evolution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Right: Feature selection frequency (across final pop)
    feature_counts = ga.best_chromosome
    colors = ['#1D9E75' if f else '#E24B4A' for f in feature_counts]
    axes[1].bar(feature_names, feature_counts, color=colors, edgecolor='white')
    axes[1].set_title('Best Chromosome — Selected Features')
    axes[1].set_xlabel('Feature')
    axes[1].set_ylabel('Selected (1) / Excluded (0)')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].set_ylim(0, 1.3)
    for i, v in enumerate(feature_counts):
        axes[1].text(i, v + 0.05, ('✓' if v else '✗'), ha='center', fontsize=12)

    plt.tight_layout()
    plt.savefig('ga_project/step3_evolution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nPlot saved: step3_evolution.png")


# ── Comparison Plot ───────────────────────────────────────────
def plot_comparison(baseline_f1, ga_f1, n_ga_features):
    fig, ax = plt.subplots(figsize=(7, 5))
    labels  = [f'Baseline\n(13 features)', f'GA Selected\n({n_ga_features} features)']
    values  = [baseline_f1, ga_f1]
    colors  = ['#3B8BD4', '#1D9E75']
    bars    = ax.bar(labels, values, color=colors, width=0.4, edgecolor='white')

    ax.set_ylim(0, 1.0)
    ax.set_ylabel('F1 Score (macro)')
    ax.set_title('Baseline vs GA Feature Selection', fontweight='bold')
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.01,
                f'{val:.4f}', ha='center', fontweight='bold', fontsize=12)
    plt.tight_layout()
    plt.savefig('ga_project/step3_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Plot saved: step3_comparison.png")


# ── Run if executed directly (after GA is done) ───────────────
if __name__ == "__main__":
    print("\nRunning fitness function test with a sample chromosome...")
    test_chrom = np.array([1,1,0,1,0,1,0,1,0,1,0,1,1])
    f1 = fitness_function(test_chrom)
    print(f"Sample chromosome F1: {f1:.4f}")
    print("\nStep 3 fitness module — OK")
    print("(Full evaluation runs in step4_main.py after GA completes)")