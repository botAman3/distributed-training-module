import os
from pathlib import Path
from torchvision.datasets import MNIST
from torchvision import transforms
from PIL import Image

def save_as_jpeg(dataset, split="train"):
    base_dir = Path(f"mnist/{split}")
    transform = transforms.ToPILImage()
    
    for i, (img, label) in enumerate(dataset):
        label_dir = base_dir / str(label)
        label_dir.mkdir(parents=True, exist_ok=True)

        img_path = label_dir / f"{i:05d}.jpg"
        if not isinstance(img, Image.Image):
            img = transform(img)
        img.convert('L').save(img_path)  # Convert to grayscale before saving

def main():
    # Create directories if they don't exist
    Path("mnist/train").mkdir(parents=True, exist_ok=True)
    Path("mnist/test").mkdir(parents=True, exist_ok=True)

    # Download and save training data
    train_data = MNIST(root="./", train=True, download=True)
    save_as_jpeg(train_data, split="train")
    print(f"Saved {len(train_data)} training images")

    # Download and save test data
    test_data = MNIST(root="./", train=False, download=True)
    save_as_jpeg(test_data, split="test")
    print(f"Saved {len(test_data)} test images")

if __name__ == "__main__":
    main()