import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from torchvision.transforms import v2 as transforms
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt

data = Path("./Cyrillic")
save_path = Path(__file__).parent

class ImageDataSet(Dataset):
    def __init__(self, path, train=True, train_ratio=0.75, augment=True):
        self.path = path
        self.train = train
        self.augment = augment
        self.data = []  # изображения
        self.labels = []  # метки
        self.class_names = []

        self._load_data()
        self._split_data(train_ratio)

    def _load_data(self):
        for class_idx, class_dir in enumerate(sorted(self.path.glob("*"))):
            class_name = class_dir.name
            self.class_names.append(class_name)
            for img_path in class_dir.glob("*.png"):
                img = imread(img_path)[:,:,3]
                binary = (img>0).astype(np.uint8)
                self.data.append(binary)
                self.labels.append(class_idx)
    def _split_data(self, train_ratio):
        indices = np.random.permutation(len(self.data))
        split_idx = int(train_ratio * len(self.data))
        
        self.train_indices = indices[:split_idx]
        self.test_indices = indices[split_idx:]
        
        if self.train:
            self.indices = self.train_indices
        else:
            self.indices = self.test_indices

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        image_np = self.data[actual_idx]              

        image = torch.from_numpy(image_np).unsqueeze(0).float() 

        transform = self._get_transform()
        image = transform(image)

        return image, self.labels[actual_idx]
    
    def _get_transform(self):
        base = [
        transforms.Resize((32, 32), antialias=True),
    ]

        if self.train and self.augment:
            train_transform = transforms.Compose([
                *base,
                transforms.RandomAffine(12, (0.12, 0.12),(0.75, 1.25),8),
                transforms.RandomPerspective(distortion_scale=0.2, p=0.4),
                transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
            return train_transform
        else:
            return transforms.Compose([
                *base,
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
        
class CyrrillicNet(nn.Module):
    def __init__(self):
        super(CyrrillicNet, self,).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2,2)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2,2)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu4 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 4 * 4,512)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512,34)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu4(x)
        x = self.pool3(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


if __name__ == "__main__":
    train_data = ImageDataSet(data,train=True,train_ratio=0.75,augment=True)
    test_data = ImageDataSet(data,train=False,train_ratio=0.75,augment=False)


    train_loader = DataLoader(train_data, 
                              batch_size=64,
                              shuffle=True)
    
    
    test_loader = DataLoader(test_data, 
                              batch_size=64,
                              shuffle=False)
    
    model = CyrrillicNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    train_loss = []
    train_acc = []
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


    import time
    t = time.perf_counter()
    for epoch in range(25):
        model.train()
        run_loss = 0.0
        total = 0
        correct = 0
        for batch_idx, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            run_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        scheduler.step()
        loss = run_loss / len(train_loader)
        acc = 100 * (correct / total)
        train_loss.append(loss)
        train_acc.append(acc)
        print(f"Epoch {epoch + 1}, Loss: {loss}, Accuracy: {acc:.3f}%")
    
    torch.save(model.state_dict(),save_path / "mnist.pth")

    plt.plot(train_acc,label="acc",color = "blue")
    plt.plot(train_loss,label="loss", color = "red")
    plt.legend()
    plt.savefig('train.png')

    test_data_for_inference = {
        'images': [],
        'labels': [],
        'class_names': test_data.class_names
    }
    
    for i in range(len(test_data)):
        img, label = test_data[i]
        test_data_for_inference['images'].append(img)
        test_data_for_inference['labels'].append(label)
    
    
    test_data_for_inference['images'] = torch.stack(test_data_for_inference['images'])
    test_data_for_inference['labels'] = torch.tensor(test_data_for_inference['labels'])
    
    torch.save(test_data_for_inference, save_path / "test_data.pt")
    

    