# ============================================================
# STEP 1: DATA LOADING & PREPROCESSING
# Member 1's responsibility
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ── 1. Load Dataset ──────────────────────────────────────────
df = pd.read_csv('ga_project/raw_dataset.csv')

print("=" * 50)
print("DATASET LOADED")
print("=" * 50)
print(f"Shape: {df.shape}")
print(f"\nFirst 5 rows:\n{df.head()}")
print(f"\nMissing values:\n{df.isnull().sum()}")

# ── 2. Handle Missing Values ─────────────────────────────────
df['ca'].fillna(df['ca'].median(), inplace=True)
df['thal'].fillna(df['thal'].mode()[0], inplace=True)

print(f"\nAfter handling missing values: {df.isnull().sum().sum()} nulls remain")

# ── 3. Encode Target (0 = No disease, 1 = Disease) ──────────
df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)
print(f"\nTarget distribution:\n{df['target'].value_counts()}")

# ── 4. Separate Features & Target ────────────────────────────
X = df.drop('target', axis=1)
y = df['target']

feature_names = list(X.columns)
print(f"\nFeatures ({len(feature_names)}): {feature_names}")

# ── 5. Scale Features ────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=feature_names)

# ── 6. EDA Plots ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('UCI Heart Disease - Exploratory Data Analysis', fontsize=15, fontweight='bold')

# Plot 1: Target distribution
axes[0, 0].bar(['No Disease (0)', 'Disease (1)'],
               df['target'].value_counts().values,
               color=['#5DCAA5', '#E24B4A'], edgecolor='white', linewidth=0.5)
axes[0, 0].set_title('Target Distribution')
axes[0, 0].set_ylabel('Count')
for i, v in enumerate(df['target'].value_counts().values):
    axes[0, 0].text(i, v + 1, str(v), ha='center', fontweight='bold')

# Plot 2: Age distribution by target
df.groupby('target')['age'].plot(kind='kde', ax=axes[0, 1])
axes[0, 1].set_title('Age Distribution by Target')
axes[0, 1].set_xlabel('Age')
axes[0, 1].legend(['No Disease', 'Disease'])

# Plot 3: Correlation heatmap
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn',
            ax=axes[1, 0], cbar_kws={'shrink': 0.8}, annot_kws={'size': 7})
axes[1, 0].set_title('Feature Correlation Heatmap')

# Plot 4: Feature means by target
feature_means = df.groupby('target')[feature_names[:6]].mean()
feature_means.T.plot(kind='bar', ax=axes[1, 1], color=['#5DCAA5', '#E24B4A'])
axes[1, 1].set_title('Feature Means by Target (first 6)')
axes[1, 1].set_xlabel('Feature')
axes[1, 1].set_ylabel('Mean Value')
axes[1, 1].legend(['No Disease', 'Disease'])
axes[1, 1].tick_params(axis='x', rotation=30)

plt.tight_layout()
plt.savefig('ga_project/step1_eda.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nEDA plot saved: step1_eda.png")

# ── 7. Save Cleaned Data ─────────────────────────────────────
X_scaled.to_csv('ga_project/X_scaled.csv', index=False)
y.to_csv('ga_project/y.csv', index=False)
df.to_csv('ga_project/cleaned_data.csv', index=False)

print("\nSaved: X_scaled.csv, y.csv, cleaned_data.csv")
print("\nStep 1 Complete!")