import cv2 
import numpy as np
from pathlib import Path
from train_model import CyrrillicNet
from torch.utils.data import TensorDataset,DataLoader
import torch

model_path = Path(__file__).parent/"mnist.pth"
data_path = Path(__file__).parent/"test_data.pt"

test_data = torch.load(data_path)
test_images = test_data['images']
test_labels = test_data['labels']
test_dataset = TensorDataset(test_images, test_labels)


test_loader = DataLoader(test_dataset, 
                              batch_size=64,
                              shuffle=False)

model = CyrrillicNet()
model.load_state_dict(torch.load(model_path))
model.eval()

test_correct = 0
test_total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

test_acc = 100 * test_correct / test_total
print(f"Accuracy: {test_acc}")