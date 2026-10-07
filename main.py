import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedShuffleSplit

from dataset import AlbumentationsDataset, get_train_transforms, get_val_transforms
from model import build_efficientnet_b4
from trainer import fit

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "Chickens_RAW")
SAVE_PATH = os.path.join(BASE_DIR, "checkpoints")


def run_experiment(train_subset, val_subset, class_weights, num_classes,
                   device, batch_size, lr, dropout, run_id):
    print(f"\n==========================================")
    print(f"Deney {run_id}: BS={batch_size} | LR={lr} | Dropout={dropout}")
    print(f"==========================================")

    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=torch.cuda.is_available()
    )

    model = build_efficientnet_b4(num_classes=num_classes, dropout_rate=dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=50,
        save_path=SAVE_PATH,
        run_id=run_id
    )
    del model, optimizer, scheduler, train_loader, val_loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == '__main__':
    os.makedirs(SAVE_PATH, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Cihaz: {device} | Veri Yolu: {DATA_DIR}")

    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"'{DATA_DIR}' dizini bulunamadı. Lütfen klasörü kontrol edin.")

    # Veri hazırlığı ve sınıf ağırlıkları
    base_dataset = AlbumentationsDataset(root=DATA_DIR)
    labels = base_dataset.targets
    num_classes = len(base_dataset.classes)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(sss.split(np.zeros(len(labels)), labels))

    train_dataset = AlbumentationsDataset(DATA_DIR, alb_transform=get_train_transforms())
    val_dataset = AlbumentationsDataset(DATA_DIR, val_transform=get_val_transforms())

    train_subset = Subset(train_dataset, train_idx)
    val_subset = Subset(val_dataset, val_idx)

    train_labels = [labels[i] for i in train_idx]
    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    # Hiperparametre Izgarası
    batch_sizes = [4, 8, 16]
    learning_rates = [1e-5, 3e-5, 1e-4, 3e-4]
    dropout_rates = [0.3, 0.4, 0.5]

    run_id = 0
    for bs in batch_sizes:
        for lr in learning_rates:
            for drop in dropout_rates:
                run_id += 1
                run_experiment(
                    train_subset, val_subset, class_weights, num_classes,
                    device, bs, lr, drop, run_id
                )