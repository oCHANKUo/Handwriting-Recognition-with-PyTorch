import os
import torch
from PIL import Image, ImageOps
from torchvision.transforms import ToTensor
import time

# Device Configuration - Windows CPU optimized
def get_device():
    """
    Returns CPU device for Windows.
    MPS check removed (Apple Silicon only).
    CUDA check kept in case NVIDIA GPU is available.
    """
    if torch.cuda.is_available():
        print("CUDA GPU detected! Using GPU for training.")
        return torch.device("cuda")
    else:
        print("Using CPU for training.")
        return torch.device("cpu")

# Model Save/Load
def save_model(model, file_path):
    """Saves model weights to disk."""
    torch.save(model.state_dict(), file_path)
    print(f"Model saved to {file_path}")

def load_model(model, file_path, device):
    """Loads model weights from disk and moves to specified device."""
    model.load_state_dict(torch.load(file_path, map_location=device, weights_only=True))
    model.to(device)
    print(f"Model loaded from {file_path}")
    return model

# Image Preprocessing
def preprocess_image(image_path, device):
    """
    Preprocesses image for model inference:
    - Converts to grayscale
    - Inverts colors (white background -> black background)
    - Converts to tensor
    - Adds batch dimension
    - Moves to device
    """
    with ImageOps.invert(Image.open(image_path).convert('L')) as img:
        img_tensor = ToTensor()(img).unsqueeze(0).to(device)
    return img_tensor

# Metrics
def calculate_accuracy(y_pred, y_true):
    """Calculates classification accuracy."""
    correct = (y_pred.argmax(dim=1) == y_true).sum().item()
    total = y_true.size(0)
    return correct / total

def print_training_progress(epoch, loss, accuracy):
    """Prints formatted training progress."""
    print(f"Epoch: {epoch} | Loss: {loss:.4f} | Accuracy: {accuracy:.2%}")

# Timer Class
class Timer:
    """Context manager for timing code blocks."""
    def __enter__(self):
        self.start = time.time()
        return self
    
    def __exit__(self, *args):
        self.end = time.time()
        print(f"Time elapsed: {self.end - self.start:.2f} seconds")

# File Management
def create_directory(directory_path):
    """Creates directory if it doesn't exist."""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Created directory: {directory_path}")

def list_images_in_directory(directory_path, extensions=(".png", ".jpg", ".jpeg")):
    """Returns list of image file paths in directory."""
    return [os.path.join(directory_path, f) 
            for f in os.listdir(directory_path) if f.endswith(extensions)]