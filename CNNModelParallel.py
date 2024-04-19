import torch
import torch.nn as nn

class CNNModelParallel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModelParallel, self).__init__()

        # Define the layers to be placed on different GPUs
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1).cuda(0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1).cuda(1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1).cuda(1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 128).cuda(0)
        self.fc2 = nn.Linear(128, num_classes).cuda(0)

    def forward(self, x):
        # Run the forward pass sequentially on each GPU
        x = self.pool(torch.relu(self.conv1(x.cuda(0))))
        x = self.pool(torch.relu(self.conv2(x.cuda(1))))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4)  # Flatten feature maps
        x = torch.relu(self.fc1(x.cuda(0)))
        x = self.fc2(x)
        return x
