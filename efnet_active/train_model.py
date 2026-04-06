import torch
from torchvision import datasets, transforms
import cv2
from torch import nn, optim
from pathlib import Path
import torchvision
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time
from collections import deque


save_path = Path(__file__).parent
model_path = save_path/"model.pth"

ACC = []
LOSS = []



def build_model():
    weights = torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1
    model = torchvision.models.efficientnet_b0(weights=weights)

    for param in model.features.parameters():
        param.requires_grad = False

    features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(features,1)
    if model_path.exists():
        model.load_state_dict(torch.load(model_path))

    return model

model = build_model()
print(model)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=0.0001
)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def train(buffer):
    if len(buffer) < 10:
        return None, None
    model.train()
    images, labels = buffer.get_batch()
    optimizer.zero_grad()
    predictions = model(images).squeeze(1)
    loss = criterion(predictions, labels)
    loss.backward()
    optimizer.step()


    with torch.no_grad():
        probs = torch.sigmoid(predictions)
        predicted_labels = (probs > 0.5).float()
        accuracy = (predicted_labels == labels).float().mean().item()
    return loss.item(), accuracy


def predict(frame):
    model.eval()
    tensor = transform(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    tensor = tensor.unsqueeze(0)
    with torch.no_grad():
        predicted = model(tensor).squeeze()
        prob = torch.sigmoid(predicted).item()
    label = "person" if prob > 0.5 else "no person"
    return label, prob


class Buffer():
    def __init__(self, maxsize=16):
        self.frames = deque(maxlen=maxsize)
        self.labels = deque(maxlen=maxsize)
    
    def append(self, tensor, label):
        self.frames.append(tensor)
        self.labels.append(label)

    def __len__(self):
        return len(self.frames)
    
    def get_batch(self):
        images = torch.stack(list(self.frames))
        labels = torch.tensor(list(self.labels),dtype=torch.float32)

        return images, labels

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    buffer = Buffer()
    count_labeled = 0
    while True:
        _, frame = cap.read()
        cv2.imshow("Camera", frame)
        key = cv2.waitKey(1) & 0xFF
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if key == ord("q"):
            fig, axs = plt.subplots(1, 2)
            axs[0].plot(ACC, color="blue")
            axs[0].set_title("ACC")
            
            axs[1].plot(LOSS, color="red")
            axs[1].set_title("LOSS")
            plt.savefig('train.png')
            # plt.show()
            break
        elif key == ord("1"): # person
            tensor = transform(image)
            buffer.append(tensor, 1.0)
            count_labeled += 1
        elif key == ord("2"): # no person
            tensor = transform(image)
            buffer.append(tensor, 0.0)
            count_labeled += 1
        elif key == ord("p"):
            t = time.perf_counter()
            label, confidence = predict(frame)
            print(time.perf_counter()-t)
            print(label, confidence)
        elif key == ord("s"): # save model
            torch.save(model.state_dict(),save_path / "model.pth")


        print(count_labeled)
        if count_labeled >= buffer.frames.maxlen:
            loss, accuracy = train(buffer)
            LOSS.append(loss)
            ACC.append(accuracy)
            if loss:
                print(f"Loss = {loss}")
            count_labeled = 0