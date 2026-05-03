from train import (Decoder, Encoder, ImageDataset)
import matplotlib.pyplot as plt
import torch

row_titles = [
    "Фикс. текст, случайная позиция",
    "Случ. текст, фикс. длина, фикс. позиция",
    "Случ. текст, случ. длина, фикс. позиция",
    "Случ. текст, случ. длина, случ. позиция",
]

enc_ft_rp = Encoder()
enc_rt_fl_fp = Encoder()
enc_rt_rl_fp = Encoder()
enc_rt_rl_rp = Encoder()

dec_ft_rp = Decoder()
dec_rt_fl_fp = Decoder()
dec_rt_rl_fp = Decoder()
dec_rt_rl_rp = Decoder()

enc_ft_rp.load_state_dict(torch.load("./models/enc_ft_rp.pth"))
enc_rt_fl_fp.load_state_dict(torch.load("./models/enc_rt_fl_fp.pth"))
enc_rt_rl_fp.load_state_dict(torch.load("./models/enc_rt_rl_fp.pth"))
enc_rt_rl_rp.load_state_dict(torch.load("./models/enc_rt_rl_rp.pth"))

dec_ft_rp.load_state_dict(torch.load("./models/dec_ft_rp.pth"))
dec_rt_fl_fp.load_state_dict(torch.load("./models/dec_rt_fl_fp.pth"))
dec_rt_rl_fp.load_state_dict(torch.load("./models/dec_rt_rl_fp.pth"))
dec_rt_rl_rp.load_state_dict(torch.load("./models/dec_rt_rl_rp.pth"))

dataset = ImageDataset(10, 256)
image, _ = dataset[1]



with torch.no_grad():

    latent_1 = enc_ft_rp(image.unsqueeze(0))
    latent_2 = enc_rt_fl_fp(image.unsqueeze(0))
    latent_3 = enc_rt_rl_fp(image.unsqueeze(0))
    latent_4 = enc_rt_rl_rp(image.unsqueeze(0))

    result_1 = dec_ft_rp(latent_1)
    result_2 = dec_rt_fl_fp(latent_2)
    result_3 = dec_rt_rl_fp(latent_3)
    result_4 = dec_rt_rl_rp(latent_4)

    fig, axes = plt.subplots(4, 3, figsize=(12, 16))
    

    axes[0, 0].imshow(image.squeeze().cpu().numpy())
    axes[0, 1].imshow(result_1.squeeze().cpu().detach().numpy())
    axes[0, 2].imshow(image.squeeze() - result_1.squeeze())
    
    axes[1, 0].imshow(image.squeeze().cpu().numpy())
    axes[1, 1].imshow(result_2.squeeze().cpu().detach().numpy())
    axes[1, 2].imshow(image.squeeze() - result_2.squeeze())
    
    axes[2, 0].imshow(image.squeeze().cpu().numpy())
    axes[2, 1].imshow(result_3.squeeze().cpu().detach().numpy())
    axes[2, 2].imshow(image.squeeze() - result_3.squeeze())
    
    axes[3, 0].imshow(image.squeeze().cpu().numpy())
    axes[3, 1].imshow(result_4.squeeze().cpu().detach().numpy())
    axes[3, 2].imshow(image.squeeze() - result_4.squeeze())
    
    for ax in axes.flat:
        ax.axis('off')
    for i, title in enumerate(row_titles):
        axes[i, 0].set_title(title, fontsize=11, fontweight='bold', pad=10)    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()