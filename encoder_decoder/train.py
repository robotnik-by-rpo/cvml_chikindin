import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import numpy as np
import string

LETTERS = list(string.ascii_letters)

encoders = ["./models/enc_ft_rp.pth",
            "./models/enc_rt_fl_fp.pth",
            "./models/enc_rt_rl_fp.pth",
            "./models/enc_rt_rl_rp.pth"]

decoders = ["./models/dec_ft_rp.pth",
            "./models/dec_rt_fl_fp.pth",
            "./models/dec_rt_rl_fp.pth",
            "./models/dec_rt_rl_rp.pth"]



class ImageDataset(Dataset):
    def __init__(self, n=200, size=128, flag = None):
        super().__init__()
        self.n = n
        self.size = size
        self.flag = flag
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        self.text_func = {1: self.text_fix_pos_rand,
                          2: self.text_rand_fix_len_pos_fix,
                          3: self.text_rand_len_rand_fix_pos,
                          4: self.text_rand_len_rand_pos_rand,
                          None: self.none_text}

    def __len__(self):
        return self.n
    
    def none_text(self):
        text = "ABC"
        x, y = 30, 30
        return text, x, y

    # 1
    def text_fix_pos_rand(self):
        text = "ABC"
        x, y = np.random.randint(0, 255), np.random.randint(0, 255)
        return text, x, y

    # 2
    def text_rand_fix_len_pos_fix(self):
        x, y = 30, 30
        text = ''.join(np.random.choice(LETTERS, size=5))
        return text, x, y

    # 3
    def text_rand_len_rand_fix_pos(self):
        x, y = 30,30
        text = ''.join(np.random.choice(LETTERS, size=np.random.randint(1,10)))
        return text, x, y

    # 4
    def text_rand_len_rand_pos_rand(self):
        text = ''.join(np.random.choice(LETTERS, size=np.random.randint(1,10)))
        x, y = np.random.randint(0, 255),np.random.randint(0, 255)
        return text, x, y
        
    def __getitem__(self, idx):
        image = Image.new('L', (self.size, self.size), color=255)
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()

        text, x, y = self.text_func[self.flag]()
        
        draw.text((x,y), text, fill=0, font=font)
        tensor = self.transform(image)
        return tensor, tensor

class Encoder(nn.Module):
    def __init__(self, latent=512):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        
            nn.Conv2d(32, 64,kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128,kernel_size= 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.Conv2d(128, 256,kernel_size= 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.bottle_neck = nn.Linear(256 * 16 * 16, latent)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0),-1)
        x = self.bottle_neck(x)
        return x
    
class Decoder(nn.Module):
    def __init__(self,latent_size=512):
        super().__init__()
        self.bottle_neck = nn.Linear(latent_size, 256 * 16 * 16)
        self.features = nn.Sequential(
            nn.ConvTranspose2d(256,128,kernel_size=4, stride=2, padding=1), # Увеличение размерности 
            nn.BatchNorm2d(128),
            nn.ReLU(),  

            nn.ConvTranspose2d(128,64,kernel_size=4, stride=2, padding=1), # Увеличение размерности 
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64,32,kernel_size=4, stride=2, padding=1), # Увеличение размерности 
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(32,1,kernel_size=4, stride=2, padding=1), # Увеличение размерности 
            nn.Sigmoid(),
        )
    
    def forward(self,x):
        x = self.bottle_neck(x)
        x = x.view(x.size(0), 256, 16, 16)
        x = self.features(x)
        return x

if __name__ == "__main__":
    for i in range(1,5,1):
        print("Task", i)
        encoder = Encoder()
        decoder = Decoder()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dataset = ImageDataset(2000, 256, i)

        dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)

        encoder.to(device)
        decoder.to(device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()))

        encoder.train()
        decoder.train()

        epochs = 10
        for epoch in range(epochs):
            epoch_loss = 0.0
            for imgs, _ in dataloader:
                imgs = imgs.to(device)
                optimizer.zero_grad()
                latent = encoder(imgs)
                output = decoder(latent)
                loss = criterion(imgs, output)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            avg_loss = epoch_loss / len(dataloader)
            print(f"{epoch=}, {avg_loss=:.2f}")
            
        torch.save(encoder.state_dict(), encoders[i-1])
        torch.save(decoder.state_dict(), decoders[i-1])