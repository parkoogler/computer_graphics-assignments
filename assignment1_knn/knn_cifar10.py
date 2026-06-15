"""
Assignment 1: K-Nearest Neighbors on CIFAR-10
- Part 1: Train/Test split only
- Part 2: Train/Validation/Test split with hyperparameter tuning
- Part 3: 5-Fold Cross-Validation
- Metrics: Accuracy, Precision, Recall, F1-Score
- Plot: 5-Fold CV accuracy vs k (with error bars)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import KFold
import pickle
import os
import urllib.request
import tarfile

# ─────────────────────────────────────────
# 1. Load CIFAR-10
# ─────────────────────────────────────────

def download_cifar10(data_dir="./cifar10_data"):
    """Download and extract CIFAR-10 if not already present."""
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    tar_path = os.path.join(data_dir, "cifar-10-python.tar.gz")
    extract_path = os.path.join(data_dir, "cifar-10-batches-py")

    if not os.path.exists(extract_path):
        os.makedirs(data_dir, exist_ok=True)
        print("Downloading CIFAR-10...")
        urllib.request.urlretrieve(url, tar_path)
        print("Extracting...")
        with tarfile.open(tar_path) as f:
            f.extractall(data_dir)
        print("Done.")
    return extract_path

def load_cifar10_batch(filepath):
    with open(filepath, 'rb') as f:
        d = pickle.load(f, encoding='bytes')
    return d[b'data'], np.array(d[b'labels'])

def load_cifar10(data_dir="./cifar10_data"):
    extract_path = download_cifar10(data_dir)

    X_train_list, y_train_list = [], []
    for i in range(1, 6):
        batch_path = os.path.join(extract_path, f"data_batch_{i}")
        X, y = load_cifar10_batch(batch_path)
        X_train_list.append(X)
        y_train_list.append(y)

    X_train = np.concatenate(X_train_list, axis=0).astype(np.float32) / 255.0
    y_train = np.concatenate(y_train_list, axis=0)

    X_test, y_test = load_cifar10_batch(os.path.join(extract_path, "test_batch"))
    X_test = X_test.astype(np.float32) / 255.0

    # Flatten: 32x32x3 -> 3072
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test  = X_test.reshape(X_test.shape[0], -1)

    return X_train, y_train, X_test, y_test

CLASS_NAMES = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']

# ─────────────────────────────────────────
# 2. Metric helper
# ─────────────────────────────────────────

def report_metrics(y_true, y_pred, label=""):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec  = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1   = f1_score(y_true, y_pred, average='macro', zero_division=0)
    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1-Score : {f1:.4f}")
    print(f"{'='*50}")
    return acc, prec, rec, f1

# ─────────────────────────────────────────
# 3. Load data (using a subset for speed)
# ─────────────────────────────────────────

print("Loading CIFAR-10...")
X_train_full, y_train_full, X_test_full, y_test_full = load_cifar10()

# Use a subset for speed (KNN is O(n) at test time; full dataset is slow)
TRAIN_SIZE = 5000
TEST_SIZE  = 1000
np.random.seed(42)
train_idx = np.random.choice(len(X_train_full), TRAIN_SIZE, replace=False)
test_idx  = np.random.choice(len(X_test_full),  TEST_SIZE,  replace=False)

X_train_full = X_train_full[train_idx]
y_train_full = y_train_full[train_idx]
X_test       = X_test_full[test_idx]
y_test       = y_test_full[test_idx]

print(f"Train: {X_train_full.shape}, Test: {X_test.shape}")

# ─────────────────────────────────────────
# PART 1: Train/Test Split Only
# ─────────────────────────────────────────

print("\n" + "="*50)
print("PART 1: Train/Test Split (k=5)")
print("="*50)

k_default = 5
knn = KNeighborsClassifier(n_neighbors=k_default, n_jobs=-1)
knn.fit(X_train_full, y_train_full)
y_pred_test = knn.predict(X_test)

report_metrics(y_test, y_pred_test, label=f"Part 1 — Train/Test Split (k={k_default})")
print("\nPer-class report:")
print(classification_report(y_test, y_pred_test, target_names=CLASS_NAMES))

# ─────────────────────────────────────────
# PART 2: Train / Validation / Test Split
# ─────────────────────────────────────────

print("\n" + "="*50)
print("PART 2: Train / Validation / Test Split")
print("="*50)

# 80% train, 20% validation (from the training set)
val_split = int(0.8 * len(X_train_full))
X_train_p2 = X_train_full[:val_split]
y_train_p2 = y_train_full[:val_split]
X_val       = X_train_full[val_split:]
y_val       = y_train_full[val_split:]

k_values = [1, 3, 5, 7, 10, 15, 20]
val_accuracies = []

print("\nValidation accuracy for each k:")
for k in k_values:
    knn_val = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    knn_val.fit(X_train_p2, y_train_p2)
    val_acc = accuracy_score(y_val, knn_val.predict(X_val))
    val_accuracies.append(val_acc)
    print(f"  k={k:2d} → Val Accuracy: {val_acc:.4f}")

best_k = k_values[np.argmax(val_accuracies)]
print(f"\nBest k = {best_k} (val accuracy = {max(val_accuracies):.4f})")

# Retrain on full train set with best k, evaluate on test
knn_best = KNeighborsClassifier(n_neighbors=best_k, n_jobs=-1)
knn_best.fit(X_train_full, y_train_full)
y_pred_best = knn_best.predict(X_test)
report_metrics(y_test, y_pred_best,
               label=f"Part 2 — Best k={best_k} (Train/Val/Test)")

# ─────────────────────────────────────────
# PART 3: 5-Fold Cross-Validation
# ─────────────────────────────────────────

print("\n" + "="*50)
print("PART 3: 5-Fold Cross-Validation")
print("="*50)

k_values_cv = [1, 3, 5, 7, 10, 15, 20]
kf = KFold(n_splits=5, shuffle=True, random_state=42)

cv_mean_acc = []
cv_std_acc  = []

for k in k_values_cv:
    fold_accs = []
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train_full)):
        X_tr, X_vl = X_train_full[tr_idx], X_train_full[val_idx]
        y_tr, y_vl = y_train_full[tr_idx], y_train_full[val_idx]

        knn_cv = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
        knn_cv.fit(X_tr, y_tr)
        acc = accuracy_score(y_vl, knn_cv.predict(X_vl))
        fold_accs.append(acc)

    mean_acc = np.mean(fold_accs)
    std_acc  = np.std(fold_accs)
    cv_mean_acc.append(mean_acc)
    cv_std_acc.append(std_acc)
    print(f"  k={k:2d} → CV Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")

best_k_cv = k_values_cv[np.argmax(cv_mean_acc)]
print(f"\nBest k from CV = {best_k_cv}")

# Final evaluation on test set using best CV k
knn_cv_best = KNeighborsClassifier(n_neighbors=best_k_cv, n_jobs=-1)
knn_cv_best.fit(X_train_full, y_train_full)
y_pred_cv = knn_cv_best.predict(X_test)
report_metrics(y_test, y_pred_cv,
               label=f"Part 3 — Best CV k={best_k_cv} on Test Set")

# ─────────────────────────────────────────
# PLOT: 5-Fold CV Accuracy vs k
# ─────────────────────────────────────────

plt.figure(figsize=(8, 5))
plt.errorbar(k_values_cv, cv_mean_acc, yerr=cv_std_acc,
             fmt='-o', color='steelblue', ecolor='lightcoral',
             elinewidth=2, capsize=5, capthick=2, linewidth=2,
             label='CV Accuracy ± std')
plt.axvline(x=best_k_cv, color='red', linestyle='--', alpha=0.7,
            label=f'Best k = {best_k_cv}')
plt.xlabel('k (Number of Neighbors)', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('5-Fold Cross-Validation Accuracy vs k\n(CIFAR-10, KNN)', fontsize=13)
plt.xticks(k_values_cv)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("cv_accuracy_vs_k.png", dpi=150)
plt.show()
print("\nPlot saved to cv_accuracy_vs_k.png")
