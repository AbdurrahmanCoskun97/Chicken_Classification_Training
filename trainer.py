import os
import torch
from tqdm import tqdm


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)
        pbar.set_postfix(loss=loss.item())

    return total_loss / len(dataloader), 100.0 * correct / total


def evaluate(model, dataloader, criterion, device):
    model.eval()
    val_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

    return val_loss / len(dataloader), 100.0 * correct / total


def fit(model, train_loader, val_loader, criterion, optimizer, scheduler,
        device, epochs, save_path, run_id, patience=5, delta=0.001):
    best_val_loss = float('inf')
    counter = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch {epoch:02d}/{epochs:02d} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")

        # Early Stopping ve Ağırlık Kaydetme
        if val_loss < best_val_loss - delta:
            best_val_loss = val_loss
            counter = 0
            ckpt_name = f"run{run_id}_best.pth"
            torch.save(model.state_dict(), os.path.join(save_path, ckpt_name))
            print(f" Model güncellendi -> {ckpt_name} (Val Loss: {val_loss:.4f})")
        else:
            counter += 1
            if counter >= patience:
                print(f"⏹ Erken durdurma tetiklendi (Epoch {epoch})\n")
                break