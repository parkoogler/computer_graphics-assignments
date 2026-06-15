# Computer Vision Assignments

Student ID: 2022120035  
Name: Chanhong Park

---

## Assignment 1 — K-Nearest Neighbors on CIFAR-10

**File:** `assignment1_knn/knn_cifar10.py`

Implements KNN classification on CIFAR-10 with three evaluation strategies:

1. **Train/Test Split** — baseline with k=5
2. **Train/Validation/Test Split** — k selected by validation accuracy
3. **5-Fold Cross-Validation** — k tuned via CV, final eval on test set

Metrics reported: Accuracy, Precision, Recall, F1-Score  
Output plot: `cv_accuracy_vs_k.png`

**Run:**
```bash
cd assignment1_knn
python knn_cifar10.py
```

---

## Assignment 2 — Image Processing: HE & CLAHE

**File:** `assignment2_image_processing/image_processing.py`

Applies and compares two contrast enhancement methods:

- **HE** (Histogram Equalization) — global redistribution of pixel values
- **CLAHE** (Contrast Limited Adaptive HE) — tile-based, avoids over-amplification

Output plot: `he_clahe_comparison.png`

**Run:**
```bash
cd assignment2_image_processing
python image_processing.py
```
> Place your own image as `sample_image.jpg` in the same folder, or the script will download one automatically.

---

## Assignment 3 — Linear Classification on CIFAR-10

**File:** `assignment3_linear_classification/linear_classifier.py`

Implements a linear classifier from scratch: `f(x, W) = Wx + b`

Two loss functions implemented:
- **SVM Hinge Loss**: `L_i = Σ max(0, s_j - s_{y_i} + 1)`
- **Softmax Cross-Entropy Loss**

Optimization: Mini-batch Stochastic Gradient Descent (SGD)  
Metrics: Accuracy, Precision, Recall, F1-Score (per class + macro)  
Output plot: `linear_classifier_results.png`

**Run:**
```bash
cd assignment3_linear_classification
python linear_classifier.py
```

---

## Requirements

```
numpy
scikit-learn
matplotlib
opencv-python
```

Install with:
```bash
pip install numpy scikit-learn matplotlib opencv-python
```
