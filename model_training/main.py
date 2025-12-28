from train_model import train_model
from chess_dataset import ChessTensorDataset
from build_model import MxModel
import torch

if __name__ == "__main__":
    # Dataset paths
    train_set = ChessTensorDataset("tensor_metadata_train3.csv", "tensor_data_maia23")
    val_set = ChessTensorDataset("tensor_metadata_val3.csv", "tensor_data_maia23")
    
    # Create enhanced model with increased capacity and policy head
    model = MxModel(
        in_channels=60,   # 60 feature planes
        n_blocks=16,      # Increased from 10 to 16 blocks
        n_moves=1715,     # Number of possible chess moves
        channels=192      # Increased from 128 to 192 channels
    )
    # model = MxModel(in_channels=60, n_blocks=16, n_moves=1715, channels=192)

    
    # Use GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Train the model with improved hyperparameters
    train_model(
        model,
        train_set,
        val_set,
        batch_size=128,  # Adjust based on your GPU memory
        epochs=50,
        lr=1e-3,
        device=device,
        weight_decay=1e-4,
        save_path="mx_enhanced_model.pth"
    )
