import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from data import ModelNet40

num_points = 1024
gaussian_noise = False  # dcp에서 noise를 주는 option
unseen = True  # dcp에서 unseen object를 test할 때 쓰는 option
# dcp는 default가 False, classification을 위해서는 True를 해야된다.
factor = 4.0
batch_size = 32
test_batch_size = 10

train_loader = DataLoader(
    ModelNet40(num_points=num_points, partition='train', gaussian_noise=gaussian_noise,
               unseen=unseen, factor=factor),
    batch_size=batch_size, shuffle=True, drop_last=True)

test_loader = DataLoader(
    ModelNet40(num_points=num_points, partition='test', gaussian_noise=gaussian_noise,
               unseen=unseen, factor=factor),
    batch_size=test_batch_size, shuffle=False, drop_last=False)

print(len(train_loader), len(test_loader))