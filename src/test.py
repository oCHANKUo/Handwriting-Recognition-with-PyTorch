import torch
import sys
from model import ImageClassifier
from utils import (
    get_device, 
    load_model, 
    preprocess_image,
    predict_with_confidence,
    plot_confusion_matrix
)
from data_loader import get_data_loaders
import matplotlib.pyplot as plt
from tqdm import tqdm


def test_single_image(model_path, image_path, device):
    """
    Test model on a single image and display prediction.
    
    Args:
        model_path: path to saved model weights
        image_path: path to image file
        device: torch device
    """
    # Load model
    model = ImageClassifier(num_classes=10)
    model = load_model(model, model_path, device)
    
    # Preprocess image
    img_tensor = preprocess_image(image_path, device)
    
    # Predict
    result = predict_with_confidence(
        model, 
        img_tensor, 
        class_names=[str(i) for i in range(10)]
    )
    
    # Display result
    print(f"\n{'='*60}")
    print(f"PREDICTION RESULTS")
    print(f"{'='*60}")
    print(f"Image: {image_path}")
    print(f"Predicted Digit: {result['predicted_label']}")
    print(f"Confidence: {result['confidence']*100:.2f}%")
    print(f"\nTop 3 Predictions:")
    
    probs = result['all_probabilities']
    top3_indices = probs.argsort()[-3:][::-1]
    
    for idx in top3_indices:
        print(f"  {idx}: {probs[idx]*100:.2f}%")
    print(f"{'='*60}\n")
    
    # Visualize
    from PIL import Image
    img = Image.open(image_path).convert('L')
    plt.figure(figsize=(6, 6))
    plt.imshow(img, cmap='gray')
    plt.title(f"Predicted: {result['predicted_label']} ({result['confidence']*100:.1f}%)")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    return result


def test_on_test_set(model_path, device, batch_size=64):
    """
    Evaluate model on entire test set and generate metrics.
    
    Args:
        model_path: path to saved model weights
        device: torch device
        batch_size: batch size for evaluation
    """
    # Load model
    model = ImageClassifier(num_classes=10)
    model = load_model(model, model_path, device)
    
    # Get test data
    _, _, test_loader = get_data_loaders(batch_size=batch_size, augment=False)
    
    # Evaluate
    model.eval()
    correct = 0
    total = 0
    class_correct = [0] * 10
    class_total = [0] * 10
    
    print(f"\n{'='*60}")
    print("EVALUATING ON TEST SET")
    print(f"{'='*60}\n")
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Per-class accuracy
            for i in range(labels.size(0)):
                label = labels[i].item()
                class_correct[label] += (predicted[i] == labels[i]).item()
                class_total[label] += 1
    
    # Overall accuracy
    overall_acc = 100. * correct / total
    print(f"\n{'='*60}")
    print(f"Overall Test Accuracy: {overall_acc:.2f}%")
    print(f"Correct: {correct}/{total}")
    print(f"{'='*60}\n")
    
    # Per-class accuracy
    print("Per-Class Accuracy:")
    print("-" * 40)
    for i in range(10):
        if class_total[i] > 0:
            acc = 100. * class_correct[i] / class_total[i]
            print(f"  Digit {i}: {acc:.2f}% ({class_correct[i]}/{class_total[i]})")
    print(f"{'='*60}\n")
    
    # Generate confusion matrix
    print("Generating confusion matrix...")
    plot_confusion_matrix(
        model, 
        test_loader, 
        device, 
        class_names=[str(i) for i in range(10)]
    )
    
    return overall_acc


def batch_test_images(model_path, image_dir, device):
    """
    Test model on multiple images in a directory.
    
    Args:
        model_path: path to saved model weights
        image_dir: directory containing test images
        device: torch device
    """
    from utils import list_images_in_directory
    
    # Load model
    model = ImageClassifier(num_classes=10)
    model = load_model(model, model_path, device)
    
    # Get all images
    image_paths = list_images_in_directory(image_dir)
    
    if not image_paths:
        print(f"No images found in {image_dir}")
        return
    
    print(f"\n{'='*60}")
    print(f"BATCH TESTING ON {len(image_paths)} IMAGES")
    print(f"{'='*60}\n")
    
    results = []
    
    for img_path in tqdm(image_paths, desc="Processing images"):
        img_tensor = preprocess_image(img_path, device)
        result = predict_with_confidence(
            model, 
            img_tensor, 
            class_names=[str(i) for i in range(10)]
        )
        result['image_path'] = img_path
        results.append(result)
    
    # Display results
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    for r in results:
        filename = r['image_path'].split('/')[-1]
        print(f"{filename:30s} -> {r['predicted_label']} ({r['confidence']*100:.1f}%)")
    
    # Show confidence distribution
    confidences = [r['confidence'] for r in results]
    avg_confidence = sum(confidences) / len(confidences)
    print(f"\nAverage Confidence: {avg_confidence*100:.2f}%")
    
    # Show low confidence predictions (might be errors)
    low_conf = [r for r in results if r['confidence'] < 0.8]
    if low_conf:
        print(f"\nLow Confidence Predictions (<80%):")
        for r in low_conf:
            filename = r['image_path'].split('/')[-1]
            print(f"  {filename}: {r['predicted_label']} ({r['confidence']*100:.1f}%)")
    
    print(f"{'='*60}\n")
    
    return results


def interactive_test(model_path, device):
    """
    Interactive testing mode - keeps asking for image paths to test.
    Useful for quick testing during development.
    """
    # Load model once
    model = ImageClassifier(num_classes=10)
    model = load_model(model, model_path, device)
    
    print(f"\n{'='*60}")
    print("INTERACTIVE TESTING MODE")
    print(f"{'='*60}")
    print("Enter image path to test (or 'q' to quit)")
    print(f"{'='*60}\n")
    
    while True:
        image_path = input("Image path: ").strip()
        
        if image_path.lower() in ['q', 'quit', 'exit']:
            print("Exiting...")
            break
        
        try:
            img_tensor = preprocess_image(image_path, device)
            result = predict_with_confidence(
                model, 
                img_tensor, 
                class_names=[str(i) for i in range(10)]
            )
            
            print(f"\n  Predicted: {result['predicted_label']}")
            print(f"  Confidence: {result['confidence']*100:.2f}%\n")
            
        except FileNotFoundError:
            print(f"  Error: File not found - {image_path}\n")
        except Exception as e:
            print(f"  Error: {e}\n")


if __name__ == "__main__":
    device = get_device()
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Test on test set:        python src/test.py models/best_model.pt")
        print("  Test single image:       python src/test.py models/best_model.pt path/to/image.png")
        print("  Test directory:          python src/test.py models/best_model.pt path/to/images/ --batch")
        print("  Interactive mode:        python src/test.py models/best_model.pt --interactive")
        sys.exit(1)
    
    model_path = sys.argv[1]
    
    if len(sys.argv) == 2:
        # Default: test on test set
        test_on_test_set(model_path, device)
    
    elif len(sys.argv) == 3:
        if sys.argv[2] == "--interactive":
            interactive_test(model_path, device)
        else:
            # Test single image
            test_single_image(model_path, sys.argv[2], device)
    
    elif len(sys.argv) == 4 and sys.argv[3] == "--batch":
        # Batch test directory
        batch_test_images(model_path, sys.argv[2], device)
    
    else:
        print("Invalid arguments. Run without arguments to see usage.")

# Examples:
# python src/test.py models/best_model.pt
# python src/test.py models/best_model.pt my_digit.png
# python src/test.py models/best_model.pt test_images/ --batch
# python src/test.py models/best_model.pt --interactive