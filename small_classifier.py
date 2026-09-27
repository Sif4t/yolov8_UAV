"""
Small from-scratch CNN image classifier — trained on your OSD_NUM_DATASET
(digit/character glyphs, classes 0-9 + J, ~1170 images/class, 20x25px).

This is a plain custom CNN (no pretrained backbone) — the point is to have your own small,
understandable classifier rather than fine-tuning something off the shelf. It expects the
class-per-folder layout your dataset already has:

    OSD_NUM_DATASET/
      train/0/*.png  train/1/*.png ... train/J/*.png
      test/0/*.png   test/1/*.png  ... test/J/*.png

Usage:
    python3 small_classifier.py --data_dir OSD_NUM_DATASET --epochs 15

Note: this dataset is character/glyph classification, not scene object detection — it's a
different task from the streetscape/UAV YOLO work, useful as its own thing (e.g. reading
digits/characters off a sign, display, or plate crop) rather than something to merge into
the detection notebook.
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class SmallCNN(nn.Module):
    """~3 conv blocks + 2 FC layers. Small on purpose — the input glyphs are 20x25px,
    a deep/wide network would just overfit a dataset this size and this simple."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 20x25 -> 10x12

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 10x12 -> 5x6

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def get_loaders(data_dir: str, batch_size: int):
    # Images are RGBA PNGs at 20x25 — force RGB, keep native size rather than resizing
    # (resizing a 20x25 source up just interpolates noise, not real detail).
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  # glyphs are effectively binary/mono
        transforms.ToTensor(),
    ])

    train_set = datasets.ImageFolder(f"{data_dir}/train", transform=transform)
    test_set = datasets.ImageFolder(f"{data_dir}/test", transform=transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader, train_set.classes


def train(data_dir: str, epochs: int, batch_size: int, lr: float):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader, classes = get_loaders(data_dir, batch_size)
    print(f"Classes ({len(classes)}): {classes}")

    model = SmallCNN(num_classes=len(classes)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        test_acc = evaluate(model, test_loader, device)
        print(f"epoch {epoch:2d}/{epochs}  train_loss={train_loss:.4f}  test_acc={test_acc:.4f}")

    torch.save({"model_state": model.state_dict(), "classes": classes}, "small_classifier.pt")
    print("Saved small_classifier.pt")


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        preds = model(images).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Path to OSD_NUM_DATASET")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    train(args.data_dir, args.epochs, args.batch_size, args.lr)
