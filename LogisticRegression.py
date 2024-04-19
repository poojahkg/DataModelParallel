import torch
import torchvision
import torch.nn as nn

# Define logistic regression model
class LogisticRegression(nn.Module):
    def __init__(self,num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(3 * 32 * 32, num_classes)  

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten input tensor
        output = self.linear(x)
        return output