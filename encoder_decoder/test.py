from train import (Decoder, Encoder, ImageDataset)
import matplotlib.pyplot as plt
import torch

ds = ImageDataset(2000,256,4)
plt.imshow(ds[0][0][0])
plt.show()
plt.imshow(ds[1][0][0])
plt.show()