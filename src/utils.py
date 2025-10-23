import os
import torch
from PIL import Image, ImageOps
from torchvision.transforms import ToTensor, Resize, Compose
import time
import numpy as np


# Device Configuration
def get_device():
    """
    Returns the best available device (CUDA > CPU).
    Includes helpful diagnostics.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✓ CUDA GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"  CUDA Version: {torch.version.cuda}")
        return device
    else:
        print("⚠ No GPU detected. Using CPU.")
        print("  Training will be slower. Consider installing CUDA for GPU acceleration.")
        return torch.device("cpu")


# Model Save/Load
def save_model(model, file_path):
    """Saves model weights to disk."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    torch.save(model.state_dict(), file_path)
    print(f"✓ Model saved to {file_path}")


def load_model(model, file_path, device):
    """Loads model weights from disk and moves to specified device."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Model file not found: {file_path}")
    
    model.load_state_dict(torch.load(file_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()  # Set to evaluation mode
    print(f"✓ Model loaded from {file_path}")
    return model


# Image Preprocessing for Inference
def preprocess_image(image_path, device, target_size=28):
    """
    Preprocesses image for model inference:
    - Converts to grayscale
    - Resizes to target size (28x28 for MNIST)
    - Inverts colors if needed (white background -> black background)
    - Normalizes and converts to tensor
    - Adds batch dimension
    """
    img = Image.open(image_path).convert('L')
    
    # Check if image needs inversion (white digit on black vs black digit on white)
    img_array = np.array(img)
    if img_array.mean() > 127:  # Bright background
        img = ImageOps.invert(img)
    
    # Preprocessing pipeline
    transform = Compose([
        Resize((target_size, target_size)),
        ToTensor(),
    ])
    
    img_tensor = transform(img).unsqueeze(0).to(device)
    return img_tensor


def preprocess_canvas_data(canvas_data, device, target_size=28):
    """
    Preprocesses drawing from HTML canvas for inference.
    
    Args:
        canvas_data: PIL Image or numpy array from canvas
        device: torch device
        target_size: target image size (28 for MNIST/EMNIST)
    
    Returns:
        Preprocessed tensor ready for model
    """
    if isinstance(canvas_data, np.ndarray):
        img = Image.fromarray(canvas_data).convert('L')
    else:
        img = canvas_data.convert('L')
    
    # Invert if needed
    img_array = np.array(img)
    if img_array.mean() > 127:
        img = ImageOps.invert(img)
    
    transform = Compose([
        Resize((target_size, target_size)),
        ToTensor(),
    ])
    
    return transform(img).unsqueeze(0).to(device)


# Prediction with confidence
def predict_with_confidence(model, image_tensor, class_names=None):
    """
    Makes prediction and returns class with confidence scores.
    
    Args:
        model: trained model
        image_tensor: preprocessed image tensor
        class_names: list of class names (optional)
    
    Returns:
        dict with predicted_class, confidence, and all_probabilities
    """
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = probabilities.max(1)
    
    predicted_class = predicted.item()
    confidence_score = confidence.item()
    
    result = {
        'predicted_class': predicted_class,
        'confidence': confidence_score,
        'all_probabilities': probabilities[0].cpu().numpy()
    }
    
    if class_names:
        result['predicted_label'] = class_names[predicted_class]
    
    return result


# Metrics
def calculate_accuracy(y_pred, y_true):
    """Calculates classification accuracy."""
    correct = (y_pred.argmax(dim=1) == y_true).sum().item()
    total = y_true.size(0)
    return correct / total


def print_training_progress(epoch, loss, accuracy, val_loss=None, val_accuracy=None):
    """Prints formatted training progress."""
    print(f"Epoch: {epoch} | Loss: {loss:.4f} | Acc: {accuracy:.2%}", end="")
    if val_loss is not None and val_accuracy is not None:
        print(f" | Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.2%}")
    else:
        print()


# Confusion Matrix
def plot_confusion_matrix(model, data_loader, device, class_names=None):
    """
    Generates and plots confusion matrix for model evaluation.
    Useful for identifying which digits/letters are confused.
    """
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150)
    print("✓ Confusion matrix saved as 'confusion_matrix.png'")
    plt.show()


# Timer Class
class Timer:
    """Context manager for timing code blocks."""
    def __enter__(self):
        self.start = time.time()
        return self
    
    def __exit__(self, *args):
        self.end = time.time()
        elapsed = self.end - self.start
        if elapsed < 60:
            print(f"⏱ Time elapsed: {elapsed:.2f} seconds")
        else:
            minutes = int(elapsed // 60)
            seconds = elapsed % 60
            print(f"⏱ Time elapsed: {minutes}m {seconds:.2f}s")


# File Management
def create_directory(directory_path):
    """Creates directory if it doesn't exist."""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"✓ Created directory: {directory_path}")


def list_images_in_directory(directory_path, extensions=(".png", ".jpg", ".jpeg")):
    """Returns list of image file paths in directory."""
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"Directory not found: {directory_path}")
    
    return [os.path.join(directory_path, f) 
            for f in os.listdir(directory_path) if f.endswith(extensions)]


# Model info
def print_model_summary(model, input_size=(1, 1, 28, 28)):
    """
    Prints model architecture summary including parameter counts.
    """
    print("\n" + "="*60)
    print("MODEL SUMMARY")
    print("="*60)
    print(model)
    print("="*60)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")
    print("="*60 + "\n")


# Check CUDA setup
def check_cuda_setup():
    """
    Comprehensive CUDA setup check.
    Run this to diagnose GPU issues.
    """
    print("\n" + "="*60)
    print("CUDA SETUP DIAGNOSTIC")
    print("="*60)
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Compute Capability: {props.major}.{props.minor}")
            print(f"  Total Memory: {props.total_memory / 1024**3:.2f} GB")
            print(f"  Multiprocessors: {props.multi_processor_count}")
    else:
        print("\n⚠ CUDA not available. Possible reasons:")
        print("  1. No NVIDIA GPU in system")
        print("  2. CUDA toolkit not installed")
        print("  3. PyTorch installed without CUDA support")
        print("  4. GPU drivers not installed/outdated")
        print("\nTo install CUDA-enabled PyTorch:")
        print("  Visit: https://pytorch.org/get-started/locally/")
    
    print("="*60 + "\n")