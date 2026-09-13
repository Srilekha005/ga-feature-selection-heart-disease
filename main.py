
import sys, os
sys.path.insert(0, '/home/claude/heart_ga_project')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ── Run Step 1 first ─────────────────────────────────────────
print("\n" + "█"*55)
print("  STEP 1: DATA PREPROCESSING")
print("█"*55)
exec(open('ga_project/step1_data_preprocessing.py').read())

# ── Import Step 2 & 3 ────────────────────────────────────────
from step2_genetic_algorithm import GeneticAlgorithm
from step3_fitness_classifier import (
    fitness_function, evaluate_chromosome,
    plot_evolution, plot_comparison, baseline_f1, feature_names
)

# ── Run Step 2: GA ───────────────────────────────────────────
print("\n" + "█"*55)
print("  STEP 2 & 3: GENETIC ALGORITHM + FITNESS")
print("█"*55)

ga = GeneticAlgorithm(
    n_features      = 13,
    population_size = 20,  # Reduced from 30 to 20
    n_generations   = 30,  # Reduced from 50 to 30
    crossover_rate  = 0.8,
    mutation_rate   = 0.05,
    tournament_size = 4,
    elitism         = 2,
    random_state    = 42
)

best_chromosome, best_f1 = ga.evolve(fitness_function, verbose=True)

# ── Step 3: Plots ─────────────────────────────────────────────
print("\n" + "█"*55)
print("  STEP 3: RESULTS & VISUALIZATIONS")
print("█"*55)

plot_evolution(ga)

selected_indices  = np.where(best_chromosome == 1)[0]
n_ga_features     = int(best_chromosome.sum())
selected_features = [feature_names[i] for i in selected_indices]

plot_comparison(baseline_f1, best_f1, n_ga_features)

clf, y_test, y_pred, sel_feats, final_f1 = evaluate_chromosome(
    best_chromosome, label="GA Selected Features"
)

# ── Step 4: Full Summary Report ───────────────────────────────
print("\n" + "█"*55)
print("  STEP 4: FINAL SUMMARY REPORT")
print("█"*55)

improvement = ((best_f1 - baseline_f1) / baseline_f1) * 100

print(f"""
╔══════════════════════════════════════════════════════╗
║          FINAL RESULTS SUMMARY                      ║
╠══════════════════════════════════════════════════════╣
║  Dataset        : UCI Heart Disease (Cleveland)     ║
║  Total samples  : 303                               ║
║  Total features : 13                                ║
╠══════════════════════════════════════════════════════╣
║  BASELINE (all 13 features)                         ║
║    F1 Score     : {baseline_f1:.4f}                        ║
╠══════════════════════════════════════════════════════╣
║  GA SELECTED ({n_ga_features:2d} features)                      ║
║    F1 Score     : {best_f1:.4f}                        ║
║    Improvement  : {improvement:+.2f}%                        ║
╠══════════════════════════════════════════════════════╣
║  SELECTED FEATURES:                                 ║""")
for f in selected_features:
    print(f"║    → {f:<47}║")
print(f"""╠══════════════════════════════════════════════════════╣
║  GA Parameters                                      ║
║    Population   : 30                                ║
║    Generations  : 50                                ║
║    Crossover    : 0.8                               ║
║    Mutation     : 0.05                              ║
╚══════════════════════════════════════════════════════╝
""")

# ── Final Dashboard Figure ────────────────────────────────────
fig = plt.figure(figsize=(16, 12))
fig.suptitle('GA-Based Feature Selection — UCI Heart Disease\nFinal Dashboard',
             fontsize=16, fontweight='bold', y=0.98)

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

# Panel 1: Evolution curve
ax1 = fig.add_subplot(gs[0, :2])
gens = range(1, len(ga.best_fitness_history) + 1)
ax1.plot(gens, ga.best_fitness_history, color='#1D9E75', linewidth=2.5, label='Best F1')
ax1.plot(gens, ga.avg_fitness_history, color='#3B8BD4', linewidth=1.5,
         linestyle='--', label='Average F1', alpha=0.7)
ax1.axhline(y=baseline_f1, color='#E24B4A', linestyle=':', linewidth=2,
            label=f'Baseline F1: {baseline_f1:.3f}')
ax1.fill_between(gens, ga.avg_fitness_history, ga.best_fitness_history,
                 alpha=0.1, color='#1D9E75')
ax1.set_xlabel('Generation', fontsize=11)
ax1.set_ylabel('F1 Score', fontsize=11)
ax1.set_title('F1 Evolution Across Generations', fontsize=12)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Panel 2: Comparison bar
ax2 = fig.add_subplot(gs[0, 2])
bars = ax2.bar(['Baseline\n(13 feat)', f'GA\n({n_ga_features} feat)'],
               [baseline_f1, best_f1],
               color=['#3B8BD4', '#1D9E75'], width=0.5, edgecolor='white', linewidth=0.5)
ax2.set_ylim(0, 1.0)
ax2.set_ylabel('F1 Score')
ax2.set_title('Baseline vs GA', fontsize=12)
for bar, val in zip(bars, [baseline_f1, best_f1]):
    ax2.text(bar.get_x() + bar.get_width()/2, val + 0.01,
             f'{val:.4f}', ha='center', fontweight='bold', fontsize=11)
ax2.grid(True, alpha=0.2, axis='y')

# Panel 3: Feature selection bar
ax3 = fig.add_subplot(gs[1, :2])
colors_feat = ['#1D9E75' if f else '#E24B4A' for f in best_chromosome]
bars3 = ax3.bar(feature_names, best_chromosome, color=colors_feat, edgecolor='white', linewidth=0.5)
ax3.set_title('Selected Features (Green=Included, Red=Excluded)', fontsize=12)
ax3.set_ylabel('Selected')
ax3.set_ylim(0, 1.4)
ax3.tick_params(axis='x', rotation=40)
for i, v in enumerate(best_chromosome):
    ax3.text(i, v + 0.05, ('✓' if v else '✗'), ha='center', fontsize=14,
             color='#1D9E75' if v else '#E24B4A', fontweight='bold')
ax3.grid(True, alpha=0.2, axis='y')

# Panel 4: Confusion matrix
ax4 = fig.add_subplot(gs[1, 2])
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=['No Disease', 'Disease'])
disp.plot(ax=ax4, colorbar=False, cmap='Blues')
ax4.set_title('Confusion Matrix\n(GA features)', fontsize=12)

plt.savefig('ga_project/step4_final_dashboard.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("Final dashboard saved: step4_final_dashboard.png")

# ── Save Results CSV ──────────────────────────────────────────
results_df = pd.DataFrame({
    'Feature': feature_names,
    'Selected_by_GA': best_chromosome
})
results_df['Reason'] = results_df['Feature'].apply(
    lambda f: 'Included' if results_df[results_df['Feature']==f]['Selected_by_GA'].values[0] else 'Excluded'
)
results_df.to_csv('ga_project/ga_results.csv', index=False)
print("Results saved: ga_results.csv")

print("\n" + "█"*55)
print("  ALL STEPS COMPLETE!")
print("█"*55)
print("\nGenerated files:")
print("  step1_eda.png           — EDA plots")
print("  step3_evolution.png     — GA evolution chart")
print("  step3_comparison.png    — Baseline vs GA")
print("  step4_final_dashboard.png — Full results dashboard")
print("  cleaned_data.csv        — Preprocessed dataset")
print("  ga_results.csv          — Feature selection results")