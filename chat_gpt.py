import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

class ModelNet40(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = os.listdir(root_dir)
        self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            for file_name in os.listdir(class_dir):
                if file_name.endswith(".npy"):
                    self.samples.append((os.path.join(class_dir, file_name), self.class_to_idx[class_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        file_path, label = self.samples[idx]
        data = np.load(file_path).astype(np.float32)
        if self.transform:
            data = self.transform(data)
        return data, label

# Define the transformations to be applied to the data
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Define the dataset and dataloader
train_dataset = ModelNet40(root_dir="path/to/ModelNet40/train", transform=transform)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

# Iterate over the dataloader
for i, (data, labels) in enumerate(train_dataloader):
    print("Batch:", i)
    print("Data shape:", data.shape)
    print("Labels shape:", labels.shape)
