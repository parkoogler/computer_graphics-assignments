"""
Assignment 2: Image Processing — Histogram Equalization (HE) & CLAHE
- Apply HE (global histogram equalization)
- Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Display: Original | HE Result | CLAHE Result
- Display corresponding histograms
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
import os

# ─────────────────────────────────────────
# Load Image
# ─────────────────────────────────────────

IMAGE_PATH = "sample_image.jpg"

# If no local image, download a sample dark image for demonstration
if not os.path.exists(IMAGE_PATH):
    print("Downloading sample image...")
    url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/240px-PNG_transparency_demonstration_1.png"
    # Use a standard test image (Lena / baboon from USC SIPI)
    url = "https://people.sc.fsu.edu/~jburkardt/data/png/baboon.png"
    try:
        urllib.request.urlretrieve(url, IMAGE_PATH)
        print("Downloaded baboon.png as sample image.")
    except Exception:
        # Fallback: generate a synthetic dark image
        print("Creating synthetic dark image...")
        img_syn = np.zeros((400, 600), dtype=np.uint8)
        # Add some features
        cv2.circle(img_syn, (200, 200), 100, 180, -1)
        cv2.rectangle(img_syn, (350, 100), (550, 350), 120, -1)
        cv2.ellipse(img_syn, (300, 300), (80, 50), 45, 0, 360, 60, -1)
        # Make it dark by scaling down
        img_syn = (img_syn * 0.3).astype(np.uint8)
        cv2.imwrite(IMAGE_PATH, img_syn)

# ─────────────────────────────────────────
# Read & convert to grayscale
# ─────────────────────────────────────────

img_bgr = cv2.imread(IMAGE_PATH)
if img_bgr is None:
    raise FileNotFoundError(f"Cannot load image: {IMAGE_PATH}")

# Convert to grayscale for HE/CLAHE
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

print(f"Image shape: {img_gray.shape}")
print(f"Pixel value range: [{img_gray.min()}, {img_gray.max()}]")

# ─────────────────────────────────────────
# Apply HE (Global Histogram Equalization)
# ─────────────────────────────────────────

img_he = cv2.equalizeHist(img_gray)

# ─────────────────────────────────────────
# Apply CLAHE (Contrast Limited Adaptive HE)
# ─────────────────────────────────────────

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
img_clahe = clahe.apply(img_gray)

# ─────────────────────────────────────────
# Plot Results
# ─────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle("Histogram Equalization Comparison", fontsize=15, fontweight='bold')

images     = [img_gray,    img_he,       img_clahe]
titles_img = ["Original",  "HE Result",  "CLAHE Result"]
titles_his = ["Original Hist", "HE Hist", "CLAHE Hist"]

# Row 0: images
for col, (img, title) in enumerate(zip(images, titles_img)):
    axes[0, col].imshow(img, cmap='gray')
    axes[0, col].set_title(title, fontsize=13, fontweight='bold')
    axes[0, col].axis('off')

# Row 1: histograms
for col, (img, title) in enumerate(zip(images, titles_his)):
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    axes[1, col].plot(hist, color='steelblue', linewidth=1.2)
    axes[1, col].set_xlim([0, 256])
    axes[1, col].set_title(title, fontsize=12)
    axes[1, col].set_xlabel("Pixel Value")
    axes[1, col].set_ylabel("Count")
    axes[1, col].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("he_clahe_comparison.png", dpi=150, bbox_inches='tight')
plt.show()
print("Plot saved to he_clahe_comparison.png")

# ─────────────────────────────────────────
# Print statistics
# ─────────────────────────────────────────

def img_stats(img, name):
    print(f"  {name:12s} → mean={img.mean():.1f}, std={img.std():.1f}, "
          f"min={img.min()}, max={img.max()}")

print("\nImage Statistics:")
img_stats(img_gray,  "Original")
img_stats(img_he,    "HE")
img_stats(img_clahe, "CLAHE")

print("\nKey Differences:")
print("  HE    : Global equalization — redistributes ALL pixel values uniformly.")
print("          Can over-amplify noise in uniform regions.")
print("  CLAHE : Adaptive (tile-based) equalization with clip limit.")
print("          Avoids over-amplification, preserves local contrast better.")
