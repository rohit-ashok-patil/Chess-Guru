import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time
from build_model import MxModel, print_trainable_params

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.early_stop

def train_model(model, train_dataset, val_dataset, batch_size=128, epochs=20, lr=1e-3, device='cpu', weight_decay=0, save_path="mx_best_model.pth"):
    print_trainable_params(model)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Debug: check tensor shapes
    sample_input, _ = next(iter(train_loader))
    print(f"Input tensor shape: {sample_input.shape}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    
    model.to(device)

    early_stopping = EarlyStopping(patience=8, min_delta=0.001)

    best_val_loss = float('inf')
    best_epoch = -1

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        start_time = time.time()
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} Training", leave=False)
        for inputs, labels in train_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            train_bar.set_postfix(loss=loss.item())
        epoch_loss = running_loss / len(train_dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} Validation", leave=False)
            for inputs, labels in val_bar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                val_bar.set_postfix(loss=loss.item(), accuracy=correct/total)
        val_loss /= len(val_dataset)
        val_acc = correct / total
        
        # Update learning rate based on validation loss
        scheduler.step(val_loss)
        
        elapsed = time.time() - start_time
        eta = elapsed * (epochs - epoch - 1)
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {epoch_loss:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f} - ETA: {eta:.1f}s")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), save_path)
            print(f"Best model saved at epoch {best_epoch} with val loss {best_val_loss:.4f}")

        # Early stopping
        if early_stopping(val_loss):
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best model was at epoch {best_epoch} with val loss {best_val_loss:.4f}.")
    print(f"Model saved to {save_path}")
