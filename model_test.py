import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data import ModelNet40

class PointNet(nn.Module):

    def __init__(self, k=40, num_of_points=1024, is_training=True):
        super().__init__()

        self.is_training = is_training
        self.num_of_points = num_of_points
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(128, 1024, kernel_size=1, bias=False)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)


    def forward(self, x):
        # x: num_of_objects * 3 * num_point
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]  # TODO Need to understand
        x = x.view(-1, 1024)  # TODO Need to understand
        x = F.relu(self.bn4(self.fc1(x)))

        x = F.dropout(x, 0.5, training=self.is_training)
        x = F.relu(self.bn5(self.fc2(x)))
        x = F.dropout(x, 0.5, training=self.is_training)
        x = self.fc3(x)

        return x


if __name__ == "__main__":
    model = PointNet(40, num_of_points=512, is_training=False)
    model.load_state_dict(torch.load("model_3_27_20_18.pth"))

    test_data = DataLoader(ModelNet40("test", 1024, 50))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    model = model.to(device)

    model.eval()

    with torch.no_grad():
        for pcls, lbls in test_data:
            pcls = pcls.permute(0, 2, 1)
            pcls, lbls = pcls.to(device), lbls.to(device)
            pred = model(pcls)

            for num, i in enumerate(pred):
                print(f"predicted: {i.argmax(0)}, Ground Truth: {lbls[num]}")
