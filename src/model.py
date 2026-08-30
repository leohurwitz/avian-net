import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from torchvision.models import resnet18, ResNet18_Weights

# 3-Layer CNN Architecture
class AvianNetModelV1(nn.Module):

    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        
        super().__init__()
    
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, # input_shape is 3 as there's 3 color channels
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1), 
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv_block_2 = nn.Sequential(
                    nn.Conv2d(in_channels=hidden_units, # input_shape is 3 as there's 3 color channels
                              out_channels=hidden_units,
                              kernel_size=3,
                              stride=1,
                              padding=1), 
                    nn.ReLU(),
                    nn.Conv2d(in_channels=hidden_units,
                              out_channels=hidden_units,
                              kernel_size=3,
                              stride=1,
                              padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2)
                )
        

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features= hidden_units*32*54,
                      out_features=output_shape)
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.classifier(x)
        return x

# Transfer Learning using ResNet18
class AvianNetModelV2(nn.Module):
    def __init__(self, num_classes, freeze_weights=True):
        super().__init__()

        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)

        if freeze_weights == True:
            for param in self.backbone.parameters():
                    param.requires_grad == False

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
         return self.backbone(x)



