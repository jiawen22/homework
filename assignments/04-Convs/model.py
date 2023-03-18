import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(torch.nn.Module):
    """
    The CNN model
    """

    def __init__(self, num_channels: int, num_classes: int) -> None:
        """
        Initialize the model
        """
        super().__init__()
        self.conv_layer1 = nn.Conv2d(num_channels, 16, 3, 2)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.batchnorm = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(784, 35)
        self.fc2 = nn.Linear(35, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Train the model
        Return:
            Output: torch.Tensor
        """
        out = F.relu(self.conv_layer1(x))
        out = self.batchnorm(out)
        out = self.max_pool(out)

        # out = out.reshape(out.size(0), -1)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)

        return out
