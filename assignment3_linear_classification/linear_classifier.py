"""
Assignment 3: Linear Classification on CIFAR-10
- Model: f(x, W) = Wx + b
- Loss 1: SVM Hinge Loss (multi-class)
- Loss 2: Softmax Cross-Entropy Loss
- Optimization: Mini-batch Gradient Descent
- Metrics: Accuracy, Precision, Recall, F1-Score
- Plots: Training loss curve, accuracy per class
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score, classification_report)
import pickle, os, urllib.request, tarfile

# ─────────────────────────────────────────
# 1. Load CIFAR-10
# ─────────────────────────────────────────

def download_cifar10(data_dir="./cifar10_data"):
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    tar_path = os.path.join(data_dir, "cifar-10-python.tar.gz")
    extract_path = os.path.join(data_dir, "cifar-10-batches-py")
    if not os.path.exists(extract_path):
        os.makedirs(data_dir, exist_ok=True)
        print("Downloading CIFAR-10...")
        urllib.request.urlretrieve(url, tar_path)
        with tarfile.open(tar_path) as f:
            f.extractall(data_dir)
        print("Done.")
    return extract_path

def load_cifar10(data_dir="./cifar10_data"):
    path = download_cifar10(data_dir)
    Xs, ys = [], []
    for i in range(1, 6):
        with open(os.path.join(path, f"data_batch_{i}"), 'rb') as f:
            d = pickle.load(f, encoding='bytes')
        Xs.append(d[b'data']); ys.append(np.array(d[b'labels']))
    X_train = np.concatenate(Xs).astype(np.float32) / 255.0
    y_train = np.concatenate(ys)
    with open(os.path.join(path, "test_batch"), 'rb') as f:
        d = pickle.load(f, encoding='bytes')
    X_test = d[b'data'].astype(np.float32) / 255.0
    y_test = np.array(d[b'labels'])
    # Flatten: 32x32x3 -> 3072
    X_train = X_train.reshape(len(X_train), -1)
    X_test  = X_test.reshape(len(X_test), -1)
    return X_train, y_train, X_test, y_test

CLASS_NAMES = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']

print("Loading CIFAR-10...")
X_train_full, y_train_full, X_test_full, y_test_full = load_cifar10()

# Subset for speed
np.random.seed(42)
TRAIN_N, TEST_N = 10000, 1000
tr_idx  = np.random.choice(len(X_train_full), TRAIN_N, replace=False)
te_idx  = np.random.choice(len(X_test_full),  TEST_N,  replace=False)
X_train = X_train_full[tr_idx]; y_train = y_train_full[tr_idx]
X_test  = X_test_full[te_idx];  y_test  = y_test_full[te_idx]
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# Normalize: zero-mean per feature
mean = X_train.mean(axis=0)
X_train = X_train - mean
X_test  = X_test  - mean

NUM_CLASSES  = 10
NUM_FEATURES = X_train.shape[1]  # 3072

# ─────────────────────────────────────────
# 2. Linear Classifier: f(x, W) = Wx + b
# ─────────────────────────────────────────

class LinearClassifier:
    """
    Parametric linear classifier.
    f(x, W) = Wx + b
    W shape: (num_classes, num_features)
    b shape: (num_classes,)
    """
    def __init__(self, num_features, num_classes, lr=1e-3, reg=1e-4):
        self.W   = np.random.randn(num_classes, num_features) * 0.01
        self.b   = np.zeros(num_classes)
        self.lr  = lr
        self.reg = reg  # L2 regularization strength

    def scores(self, X):
        """Compute class scores: (N, C)"""
        return X @ self.W.T + self.b   # (N, C)

    def predict(self, X):
        return np.argmax(self.scores(X), axis=1)

# ─────────────────────────────────────────
# 3. SVM Hinge Loss
#    L_i = sum_{j != y_i} max(0, s_j - s_{y_i} + 1)
# ─────────────────────────────────────────

def svm_loss_and_grad(model, X, y):
    N = X.shape[0]
    S = model.scores(X)                    # (N, C)

    correct_scores = S[np.arange(N), y].reshape(-1, 1)   # (N, 1)
    margins = np.maximum(0, S - correct_scores + 1)       # (N, C)
    margins[np.arange(N), y] = 0                          # exclude correct class

    loss = margins.sum() / N
    # L2 regularization
    loss += model.reg * np.sum(model.W ** 2)

    # Gradients
    mask = (margins > 0).astype(float)
    mask[np.arange(N), y] -= mask.sum(axis=1)

    dW = (mask.T @ X) / N + 2 * model.reg * model.W     # (C, D)
    db = mask.sum(axis=0) / N                            # (C,)
    return loss, dW, db

# ─────────────────────────────────────────
# 4. Softmax Cross-Entropy Loss
#    L_i = -log( e^{s_{y_i}} / sum_j e^{s_j} )
# ─────────────────────────────────────────

def softmax_loss_and_grad(model, X, y):
    N = X.shape[0]
    S = model.scores(X)                           # (N, C)

    # Numerical stability: shift by max score
    S -= S.max(axis=1, keepdims=True)
    exp_S = np.exp(S)
    probs = exp_S / exp_S.sum(axis=1, keepdims=True)   # (N, C)

    loss = -np.log(probs[np.arange(N), y] + 1e-12).mean()
    loss += model.reg * np.sum(model.W ** 2)

    # Gradient
    dprobs = probs.copy()
    dprobs[np.arange(N), y] -= 1
    dprobs /= N

    dW = dprobs.T @ X + 2 * model.reg * model.W    # (C, D)
    db = dprobs.sum(axis=0)                         # (C,)
    return loss, dW, db

# ─────────────────────────────────────────
# 5. Mini-batch SGD Training Loop
# ─────────────────────────────────────────

def train(model, X, y, loss_fn, epochs=30, batch_size=256, verbose=True):
    N = X.shape[0]
    history = []
    for epoch in range(1, epochs + 1):
        # Shuffle
        idx = np.random.permutation(N)
        X_sh, y_sh = X[idx], y[idx]

        epoch_loss = 0.0
        num_batches = 0
        for start in range(0, N, batch_size):
            Xb = X_sh[start:start + batch_size]
            yb = y_sh[start:start + batch_size]

            loss, dW, db = loss_fn(model, Xb, yb)
            model.W -= model.lr * dW
            model.b -= model.lr * db
            epoch_loss += loss
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        history.append(avg_loss)
        if verbose and (epoch % 5 == 0 or epoch == 1):
            train_acc = accuracy_score(y, model.predict(X))
            print(f"  Epoch {epoch:3d}/{epochs} | Loss: {avg_loss:.4f} | "
                  f"Train Acc: {train_acc:.4f}")
    return history

# ─────────────────────────────────────────
# 6. Train SVM model
# ─────────────────────────────────────────

print("\n" + "="*55)
print("  Training: Linear SVM (Hinge Loss)")
print("="*55)

svm_model = LinearClassifier(NUM_FEATURES, NUM_CLASSES, lr=1e-2, reg=1e-4)
svm_history = train(svm_model, X_train, y_train,
                    loss_fn=svm_loss_and_grad, epochs=50, batch_size=256)

y_pred_svm = svm_model.predict(X_test)
print("\nSVM Test Results:")
print(f"  Accuracy : {accuracy_score(y_test, y_pred_svm):.4f}")
print(f"  Precision: {precision_score(y_test, y_pred_svm, average='macro', zero_division=0):.4f}")
print(f"  Recall   : {recall_score(y_test, y_pred_svm, average='macro', zero_division=0):.4f}")
print(f"  F1-Score : {f1_score(y_test, y_pred_svm, average='macro', zero_division=0):.4f}")
print("\nPer-class report (SVM):")
print(classification_report(y_test, y_pred_svm, target_names=CLASS_NAMES))

# ─────────────────────────────────────────
# 7. Train Softmax model
# ─────────────────────────────────────────

print("\n" + "="*55)
print("  Training: Softmax (Cross-Entropy Loss)")
print("="*55)

softmax_model = LinearClassifier(NUM_FEATURES, NUM_CLASSES, lr=1e-2, reg=1e-4)
softmax_history = train(softmax_model, X_train, y_train,
                        loss_fn=softmax_loss_and_grad, epochs=50, batch_size=256)

y_pred_sm = softmax_model.predict(X_test)
print("\nSoftmax Test Results:")
print(f"  Accuracy : {accuracy_score(y_test, y_pred_sm):.4f}")
print(f"  Precision: {precision_score(y_test, y_pred_sm, average='macro', zero_division=0):.4f}")
print(f"  Recall   : {recall_score(y_test, y_pred_sm, average='macro', zero_division=0):.4f}")
print(f"  F1-Score : {f1_score(y_test, y_pred_sm, average='macro', zero_division=0):.4f}")
print("\nPer-class report (Softmax):")
print(classification_report(y_test, y_pred_sm, target_names=CLASS_NAMES))

# ─────────────────────────────────────────
# 8. Plots
# ─────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Plot 1: Training Loss Curves
axes[0].plot(svm_history,     label='SVM Loss',      color='steelblue', linewidth=2)
axes[0].plot(softmax_history, label='Softmax Loss',  color='tomato',    linewidth=2)
axes[0].set_title('Training Loss Curve\nLinear Classifier on CIFAR-10', fontsize=12)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Per-class F1 comparison
svm_f1 = f1_score(y_test, y_pred_svm, average=None, zero_division=0)
sm_f1  = f1_score(y_test, y_pred_sm,  average=None, zero_division=0)
x = np.arange(NUM_CLASSES)
w = 0.35
axes[1].bar(x - w/2, svm_f1, w, label='SVM',     color='steelblue', alpha=0.85)
axes[1].bar(x + w/2, sm_f1,  w, label='Softmax', color='tomato',    alpha=0.85)
axes[1].set_xticks(x)
axes[1].set_xticklabels(CLASS_NAMES, rotation=35, ha='right', fontsize=9)
axes[1].set_title('Per-class F1-Score\n(Test Set)', fontsize=12)
axes[1].set_ylabel('F1-Score')
axes[1].legend()
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig("linear_classifier_results.png", dpi=150, bbox_inches='tight')
plt.show()
print("\nPlot saved to linear_classifier_results.png")
