import sys
import os
from PIL import Image, ImageOps
import torch
from torchvision.transforms import ToTensor, Resize, Compose
from model import ImageClassifier
# from train import ImageClassifier
import matplotlib.pyplot as plt

model_path = "models/model_state.pt"

def predict_image(image_path, model_path="models/model_state.pt", device="cpu"):
    # Load Model
    model = ImageClassifier().to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    # Define transforms: Resize to 28x28 to match the model input
    transform = Compose([
        Resize((28, 28)),   # Resize to 28x28
        ToTensor()
    ])

    # Load and Process the image
    with ImageOps.invert(Image.open(image_path).convert('L')) as img:
        img_tensor = transform(img).unsqueeze(0).to(device)

    # Perform Prediction
    with torch.no_grad():
        output = model(img_tensor)
        predicted_label = torch.argmax(output).item()

    # Show image with prediction
    plt.imshow(img, cmap="gray")
    plt.title(f"Predicted Label: {predicted_label}")
    plt.axis('off')
    # plt.savefig(f"{image_path}_prediction.png")
    plt.show()

    print(f"Predicted Label: {predicted_label}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python evaluate.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"Error: File '{image_path}' not found.")
        sys.exit(1) 

    predict_image(image_path)


    # Run this script
    # python src/evaluate.py digits/1.jpg --- parameter is the location of the image