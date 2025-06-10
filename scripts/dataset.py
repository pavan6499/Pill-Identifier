from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import pytesseract
import os

class PillImprintDataset(Dataset):
    def __init__(self, root_dir, transform=None, text_transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.text_transform = text_transform

        self.samples = []
        self.class_to_idx = {}

        for idx, class_name in enumerate(sorted(os.listdir(root_dir))):
            class_path = os.path.join(root_dir, class_name)
            if os.path.isdir(class_path):
                self.class_to_idx[class_name] = idx
                for file in os.listdir(class_path):
                    if file.endswith(('.png', '.jpg', '.jpeg')):
                        self.samples.append((os.path.join(class_path, file), idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # OCR
        text = pytesseract.image_to_string(Image.open(image_path))
        if self.text_transform:
            text = self.text_transform(text)

        return image, text, label
