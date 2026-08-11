import os
import shutil
from sklearn.model_selection import train_test_split

arquivos = [f for f in os.listdir("imagens") if os.path.isfile(os.path.join("imagens", f))]
train_files, val_files = train_test_split(arquivos, test_size=0.2, random_state=42)

os.makedirs("imagens/train", exist_ok=True)
os.makedirs("imagens/val", exist_ok=True)
os.makedirs("mascaras/train", exist_ok=True)
os.makedirs("mascaras/val", exist_ok=True)

for f in train_files:
    shutil.move(f"imagens/{f}", f"imagens/train/{f}")
    shutil.move(f"mascaras/{f.replace('.jpg', '.png')}", f"mascaras/train/{f.replace('.jpg', '.png')}")

for f in val_files:
    shutil.move(f"imagens/{f}", f"imagens/val/{f}")
    shutil.move(f"mascaras/{f.replace('.jpg', '.png')}", f"mascaras/val/{f.replace('.jpg', '.png')}")