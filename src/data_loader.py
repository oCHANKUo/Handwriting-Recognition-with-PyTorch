from torchvision import datasets
from torchvision.transforms import ToTensor, Compose, RandomRotation, RandomAffine
from torch.utils.data import DataLoader, random_split


def get_data_loaders(batch_size=64, augment=True, use_cuda=False):
    """
    Returns train, validation, and test data loaders.
    
    Args:
        batch_size: Number of samples per batch
        augment: Whether to apply data augmentation (rotation, scaling, shifting)
        use_cuda: Whether CUDA is available (for pin_memory optimization)
    
    Returns:
        train_loader, val_loader, test_loader
    """
    
    # Data augmentation for training (simulates kid handwriting variations)
    if augment:
        train_transform = Compose([
            RandomRotation(degrees=15),  # Kids write at angles
            RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),    # Position variations
                scale=(0.9, 1.1)         # Size variations
            ),
            ToTensor()
        ])
    else:
        train_transform = ToTensor()
    
    # No augmentation for validation/test (measure true performance)
    test_transform = ToTensor()
    
    # Download datasets
    train_data = datasets.MNIST(
        root="data", 
        train=True, 
        download=True, 
        transform=train_transform
    )
    
    test_data = datasets.MNIST(
        root="data", 
        train=False, 
        download=True, 
        transform=test_transform
    )
    
    # Split training into train (80%) and validation (20%)
    train_size = int(0.8 * len(train_data))
    val_size = len(train_data) - train_size
    train_subset, val_subset = random_split(train_data, [train_size, val_size])
    
    # Create data loaders
    # Note: num_workers > 0 can cause issues on Windows with some libraries
    # Set to 0 for Windows compatibility
    train_loader = DataLoader(
        train_subset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0,  # 0 for CPU or Windows compatibility
        pin_memory=use_cuda  # Only pin memory if using CUDA
    )
    
    val_loader = DataLoader(
        val_subset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0,
        pin_memory=use_cuda
    )
    
    test_loader = DataLoader(
        test_data, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0,
        pin_memory=use_cuda
    )
    
    print(f"Train samples: {len(train_subset)}")
    print(f"Validation samples: {len(val_subset)}")
    print(f"Test samples: {len(test_data)}")
    
    return train_loader, val_loader, test_loader