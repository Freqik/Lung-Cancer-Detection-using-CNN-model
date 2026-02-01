import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from backend.utils.label_utils import get_binary_label

class LungDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.data = []
        self.transform = transform
        for folder in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder)
            if os.path.isdir(folder_path):
                label = get_binary_label(folder)
                for file in os.listdir(folder_path):
                    if file.endswith(('.jpg', '.png')):
                        img_path = os.path.join(folder_path, file)
                        self.data.append((img_path, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert('L')  # grayscale
        if self.transform:
            image = self.transform(image)
        return image, label
