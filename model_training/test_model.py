import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
from build_model import MxModel
from chess_dataset import ChessTensorDataset
from collections import Counter
import os

def evaluate_model(model, test_dataset, batch_size=128, device='cpu'):
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    model.eval()
    model.to(device)
    correct_top1 = 0
    correct_top3 = 0
    correct_top5 = 0
    total = 0
    all_preds = []
    all_labels = []

    # Debug: check tensor shapes
    sample_input, _ = next(iter(test_loader))
    print(f"Input tensor shape: {sample_input.shape}")

    with torch.no_grad():
        for xb, yb in tqdm(test_loader, desc="Testing"):
            xb, yb = xb.to(device), yb.to(device)
            outputs = model(xb)
            _, pred_top1 = outputs.topk(1, dim=1)
            _, pred_top3 = outputs.topk(3, dim=1)
            _, pred_top5 = outputs.topk(5, dim=1)

            correct_top1 += (pred_top1.squeeze() == yb).sum().item()
            correct_top3 += sum([yb[i].item() in pred_top3[i] for i in range(yb.size(0))])
            correct_top5 += sum([yb[i].item() in pred_top5[i] for i in range(yb.size(0))])

            total += yb.size(0)
            all_preds.extend(pred_top1.squeeze().cpu().numpy())
            all_labels.extend(yb.cpu().numpy())

    top1_acc = correct_top1 / total
    top3_acc = correct_top3 / total
    top5_acc = correct_top5 / total

    print(f"\nTest set size: {total}")
    print(f"Top-1 Accuracy: {top1_acc:.4f}")
    print(f"Top-3 Accuracy: {top3_acc:.4f}")
    print(f"Top-5 Accuracy: {top5_acc:.4f}")

    # Optional: Show most common mistakes (confusion matrix for top N moves)
    try:
        from sklearn.metrics import confusion_matrix
        import matplotlib.pyplot as plt
        # Show confusion for the 20 most common true moves
        most_common = [item for item, _ in Counter(all_labels).most_common(20)]
        mask = [lbl in most_common for lbl in all_labels]
        cm = confusion_matrix(np.array(all_labels)[mask], np.array(all_preds)[mask], labels=most_common)
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, cmap='Blues')
        plt.title("Confusion Matrix (Top 20 Moves)")
        plt.xlabel("Predicted Move Index")
        plt.ylabel("True Move Index")
        plt.colorbar()
        plt.show()
    except ImportError:
        pass

if __name__ == "__main__":
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Use absolute paths
    test_csv = os.path.join(script_dir, "tensor_metadata_test3.csv")
    tensor_dir = os.path.join(script_dir, "tensor_data_maia23")
    model_path = os.path.join(script_dir, "mx_best_model3.pth")


    # Load dataset and model
    test_set = ChessTensorDataset(test_csv, tensor_dir)
    
    # Make sure to create the model with the correct number of input channels
    # This should match the number of planes in your tensor
    model = MxModel(in_channels=60, n_blocks=16, n_moves=1715, channels=192)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))

    # Evaluate
    evaluate_model(model, test_set, device='cuda' if torch.cuda.is_available() else 'cpu')
