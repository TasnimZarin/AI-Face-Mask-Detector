# 😷 AI Face Mask Detector

A deep-learning project to automatically classify face-mask usage into five categories using **Convolutional Neural Networks (CNNs)**:

- Cloth mask  
- N95/FFP2  
- N95/FFP2 with valve  
- Surgical mask  
- Without mask  

The project was developed in two phases:
- **Part I:** Dataset collection & augmentation, preprocessing, and a custom CNN trained with **5-fold Cross Validation**.  
- **Part II:** Extended experiments with **gender-based bias analysis**, dataset balancing, **10-fold Cross Validation**, and **EarlyStopping**, leading to improved and fairer results.

---

## 📚 Project Overview
- **Goal:** Detect mask types from images and study the effect of dataset imbalance and bias on performance.  
- **Pipeline:** Data collection → Augmentation & preprocessing → CNN training → Evaluation (accuracy, precision, recall, F1, confusion matrix).  
- **Key Findings:** Model performance depends heavily on data distribution. Gender balancing, 10-fold CV, and early stopping improved generalization and reduced bias.

---

## 🧾 Dataset
- Collected from multiple public datasets (Kaggle: *Medical Mask*, *Face Mask Detection*, *MAFA*, *Dataset for Mask Detection*; **MaskedFace-Net**; **FFHQ**; **IEEE DataPort WWMR-DB**) plus curated web images.  
- **Part I:** 2,081 images → +948 augmented → **3,029 total**.  
- **Part II:** Gender split → 851 male / 1,219 female. Balanced each gender to ≈280 images per class via augmentation (rotation, flip) and down-sampling:  
  - Male: 851 → 1,394  
  - Female: 1,219 → 1,377  

**Labels:**  
`0 = cloth`, `1 = N95/FFP2`, `2 = N95/FFP2 with valve`, `3 = surgical`, `4 = without mask`

---

## 🧪 Preprocessing
- Resize to **240×240**.  
- Initial grayscale preprocessing; dataset **mean/std** calculated for normalization.  
- PyTorch **Datasets/DataLoaders** used to handle splits and augmentation pipelines.  

---

## 🧠 Model
- **Custom CNN** with 5 convolutional layers.  
- Each layer followed by **Batch Normalization** + **Leaky ReLU**.  
- First layer: 3-channel RGB input (240×240).  
- Explored AlexNet baseline, but custom CNN performed better under CV.  

**Training setup**:
- Optimizer: **Adam**  
- Loss: **Cross-entropy**  
- Learning rate: `1e-3`  
- Batch size: 32 (Part I), 64 (Part II)  

---

## 📈 Results

### Part I (5-fold CV)
- Train Accuracy: ~46%  
- Test Accuracy: ~48%  
- Macro Precision/Recall/F1: ~35%  
- Best performance on **cloth mask** and **without mask** (largest classes).  
- Struggled with **N95**, **N95 w/ valve**, and **surgical** (fewer samples).

### Part II (10-fold CV + Gender Balancing + EarlyStopping)
- Balanced male/female datasets reduced bias and improved performance.  
- Results across scenarios:

| Scenario | Mixed Testset | Female Testset | Male Testset |
|----------|---------------|----------------|--------------|
| Part I (imbalanced, 5-fold CV) | ~60% | ~58% | ~55% |
| Part I retrained (balanced, 5-fold CV) | ~49% | ~50% | ~51% |
| Part II (imbalanced, 10-fold CV) | ~63% | ~66% | ~64% |
| Part II (balanced, 10-fold CV + EarlyStopping) | ~72% | ~71% | ~72% |

- Confusion matrices show improved recognition across all classes, with continued strength in **cloth** and **no mask** detection.  
- Gender bias present in Part I was **significantly reduced** in Part II.

---

## 🛠️ Tech Stack
- **Language:** Python  
- **Frameworks/Libraries:** PyTorch, torchvision, NumPy, Pandas  
- **CV/ML:** Custom CNN, BatchNorm, Leaky ReLU, 5-fold & 10-fold Cross Validation, EarlyStopping, Adam optimizer, Cross-entropy loss  
- **Data Ops:** Augmentation (rotations, flips), CSV labeling scripts, balanced splits  
- **Metrics:** Accuracy, Precision, Recall, F1, Confusion Matrix  

---

## 🗂️ Repository Structure
