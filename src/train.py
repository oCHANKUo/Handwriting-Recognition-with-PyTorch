import sys
import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model import ImageClassifier
from data_loader import get_data_loaders
from utils import get_device, save_model
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for Windows compatibility
import matplotlib.pyplot as plt
from tqdm import tqdm


def train_one_epoch(model, train_loader, optimizer, loss_fn, device):
    """Train for one epoch and return average loss and accuracy."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    # Progress bar
    pbar = tqdm(train_loader, desc="Training", leave=False)
    
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy


def validate(model, val_loader, loss_fn, device):
    """Validate model and return average loss and accuracy."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    
    avg_loss = total_loss / len(val_loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy


def train_model(epochs=20, lr=1e-3, batch_size=64, device="cpu"):
    """
    Main training loop with validation and early stopping.
    
    Args:
        epochs: Number of training epochs
        lr: Learning rate
        batch_size: Batch size for data loaders
        device: Device to train on (cpu/cuda)
    """
    print(f"{'='*60}")
    print(f"Training Configuration")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Epochs: {epochs}")
    print(f"Learning Rate: {lr}")
    print(f"Batch Size: {batch_size}")
    print(f"{'='*60}\n")
    
    # Initialize
    try:
        model = ImageClassifier(num_classes=10).to(device)
        print(f"Model parameters: {model.count_parameters():,}\n")
    except TypeError:
        # Fallback for old model without num_classes parameter
        print("⚠ Using old model architecture (without num_classes parameter)")
        print("⚠ Recommended: Update model.py with improved architecture\n")
        model = ImageClassifier().to(device)
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model parameters: {total_params:,}\n")
    
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = CrossEntropyLoss()
    
    # Learning rate scheduler (reduces LR when validation loss plateaus)
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=3
    )
    
    # Get data loaders (returns 3 loaders)
    print("Loading datasets...")
    use_cuda = device.type == 'cuda'
    train_loader, val_loader, test_loader = get_data_loaders(
        batch_size=batch_size, 
        augment=True,
        use_cuda=use_cuda
    )
    print()
    
    # Track history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0
    patience_counter = 0
    early_stop_patience = 7
    
    # Training loop
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 60)
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, loss_fn, device)
        
        # Update learning rate
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        
        if new_lr != old_lr:
            print(f"⚡ Learning rate reduced: {old_lr:.6f} -> {new_lr:.6f}")
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model(model, "models/best_model.pt")
            print(f"✓ New best model saved! (Val Acc: {val_acc:.2f}%)")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= early_stop_patience:
            print(f"\nEarly stopping triggered! No improvement for {early_stop_patience} epochs.")
            break
    
    # Final test evaluation
    print(f"\n{'='*60}")
    print("Final Evaluation on Test Set")
    print(f"{'='*60}")
    test_loss, test_acc = validate(model, test_loader, loss_fn, device)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    
    # Save final model
    save_model(model, "models/final_model.pt")
    
    # Plot training curves
    plot_training_history(history, epochs)
    
    return history


def plot_training_history(history, epochs):
    """Plot training and validation loss/accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    epochs_range = range(1, len(history['train_loss']) + 1)
    
    # Loss plot
    ax1.plot(epochs_range, history['train_loss'], 'b-', label='Train Loss')
    ax1.plot(epochs_range, history['val_loss'], 'r-', label='Val Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(epochs_range, history['train_acc'], 'b-', label='Train Accuracy')
    ax2.plot(epochs_range, history['val_acc'], 'r-', label='Val Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150)
    print(f"\n✓ Training curves saved as 'training_history.png'")
    plt.show()


if __name__ == "__main__":
    device = get_device()
    
    # Parse command line arguments
    epochs_count = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    learning_rate = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-3
    batch_size = int(sys.argv[3]) if len(sys.argv) > 3 else 64
    
    # Train model
    history = train_model(
        epochs=epochs_count, 
        lr=learning_rate,
        batch_size=batch_size,
        device=device
    )

# Run this script from the root directory:
# python src/train.py 20              # 20 epochs, default lr and batch_size
# python src/train.py 30 0.001 128    # 30 epochs, lr=0.001, batch_size=128