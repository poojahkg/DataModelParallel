import torch
import torchvision
import torch.nn as nn

# Define logistic regression model
class LogisticRegressionMP(nn.Module):
    def __init__(self,input_size,num_classes):
        super(LogisticRegressionMP, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)  

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten input tensor
        output = self.linear(x)
        return output