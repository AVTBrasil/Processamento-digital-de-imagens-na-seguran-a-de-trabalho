import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from model import UNET

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

model = UNET(in_channels=3, out_channels=2).to(DEVICE)
checkpoint = torch.load("my_checkpoint.pth.tar", map_location=DEVICE)
model.load_state_dict(checkpoint["state_dict"])
model.eval()

transform = A.Compose([
    A.Resize(height=160, width=240),
    A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
    ToTensorV2(),
])

image_path = "imagens/val/108519221292039.jpg"
image = np.array(Image.open(image_path).convert("RGB"))

augmented = transform(image=image)
input_tensor = augmented["image"].unsqueeze(0).to(DEVICE)

with torch.no_grad():
    pred = model(input_tensor)
    pred_mask = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy()

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(image)
ax[0].set_title("Original")
ax[1].imshow(pred_mask, cmap="gray")
ax[1].set_title("Predição do modelo")
plt.savefig("resultado_predicao.png")
print("Salvo: resultado_predicao.png")