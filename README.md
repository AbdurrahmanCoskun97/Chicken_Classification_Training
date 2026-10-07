# Chicken_Classification_Training

An end-to-end deep learning pipeline designed to categorize packaged poultry cuts into 5 distinct morphological classes. The system leverages an **EfficientNet-B4** backbone with full network fine-tuning and comprehensive image augmentations to handle packaging reflections, vacuum-film folds, and optical meat surface variations.

---

## Overview

Industrial chicken sorting and quality inspection require reliable visual recognition across anatomical cuts. Packaged raw poultry introduces distinct optical challenges: specular glare from plastic wrap, skin folds, moisture beads, and subtle variations in bone-to-meat ratios.

This repository trains an image classifier to accurately recognize and categorize chicken cuts across 5 classes:
- **Breasts**
- **ButterfliedDrumsticks**
- **Drumsticks**
- **WholeLeg**
- **Wings**

---

## Why Full Fine-Tuning?

Rather than freezing the convolutional backbone and only retraining the linear classifier head, this pipeline performs **full fine-tuning** across all layers of the EfficientNet-B4 architecture.

* **Domain Shift Beyond ImageNet:** ImageNet features capture natural scenes and generic everyday objects. Packaged chicken cuts contain micro-textures, moisture sheen, tissue granularity, and subcutaneous fat distribution that pre-trained generic weights cannot sufficiently isolate.
* **Surface Reflection & Artifact Invariance:** Plastic wrapping introduces specular reflections, condensation artifacts, and wrinkles. Fine-tuning intermediate convolutional layers enables the network to learn filters invariant to wrapping glare while remaining sensitive to meat anatomy beneath the film.
* **High-Resolution Granularity ($380 \times 380$):** EfficientNet-B4 operates at $380 \times 380$ px resolution. Re-aligning early-to-mid layer receptive fields ensures spatial high-frequency details (e.g., bone cuts, cartilage, muscle fibers) propagate informative gradients through the compound scaling blocks.
* **Balanced Feature Adaptation:** Using low, log-scaled learning rates with weight decay ensures the foundational feature extractors remain stable while specialized deeper layers adapt to poultry morphology.

---

## Key Features

- **EfficientNet-B4 Backbone:** Balances depth, width, and resolution using compound coefficient scaling.
- **Aggressive Augmentations:** Robust Albumentations pipeline including full rotation ($360^\circ$), motion blur, compression artifacts, coarse dropout (cutout), and hue/saturation shifts.
- **Class Imbalance Handling:** Dynamic loss weighting via `compute_class_weight('balanced')` integrated into PyTorch's `CrossEntropyLoss`.
- **System-Aware Early Stopping:** Configurable validation loss monitoring with minimal delta thresholding (`delta=0.001`) and automatic checkpoint persistence.
- **Multi-Hyperparameter Exploration:** Grid-search loop over log-scaled learning rates and dropout probabilities.

---

## Project Structure

```text
Chicken_Classification_Training/
├── Chickens_RAW/                 # Raw dataset organized by class subfolders
│   ├── Breasts/
│   ├── ButterfliedDrumsticks/
│   ├── Drumsticks/
│   ├── WholeLeg/
│   └── Wings/
├── checkpoints/                  # Saved .pth model weights
├── dataset.py                    # Albumentations transforms & ImageFolder loader
├── model.py                      # EfficientNet-B4 classifier architecture
├── trainer.py                    # Training loops, evaluation, & early stopping
├── main.py                       # Execution entrypoint & hyperparameter grid search
├── requirements.txt              # Project dependencies
├── .gitignore                    # Ignored artifacts & weight binaries
└── README.md
```

---

## Getting Started

### Prerequisites

* Python 3.10+
* NVIDIA GPU with 8GB+ VRAM recommended (CUDA 11.8 or 12.1+)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/AbdurrahmanCoskun97/Chicken_Classification_Training.git
   cd Chicken_Classification_Training
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux/macOS:
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   # Install PyTorch with CUDA support (adjust version if necessary)
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

   # Install remaining packages
   pip install -r requirements.txt
   ```

---

## Dataset Setup

Ensure your raw dataset is structured in standard `ImageFolder` format inside `Chickens_RAW/`:

```text
Chickens_RAW/
├── Breasts/
│   ├── img_001.jpg
│   └── ...
├── ButterfliedDrumsticks/
├── Drumsticks/
├── WholeLeg/
└── Wings/
```

The training script automatically executes a stratified 80/20 train-validation split to preserve class distributions across subsets.

---

## Training & Grid Search

Run the primary training pipeline:

```bash
python main.py
```

### Default Grid Configuration

| Hyperparameter | Evaluated Values |
| :--- | :--- |
| **Batch Size** | `4` *(Optimized for $380 \times 380$ on consumer GPUs)* |
| **Learning Rates** | `1e-5`, `3e-5`, `1e-4`, `3e-4` *(Logarithmic scale for fine-tuning)* |
| **Dropout Rates** | `0.3`, `0.4`, `0.5` |
| **Optimizer** | Adam (`weight_decay=1e-4`) |
| **LR Scheduler** | StepLR (`step_size=10`, `gamma=0.1`) |
| **Patience** | 5 Epochs (`delta=0.001`) |

Checkpoints are serialized to `./checkpoints/` dynamically whenever validation loss improves past the delta threshold.

---

## License

Distributed under the MIT License. See `LICENSE` for more information.