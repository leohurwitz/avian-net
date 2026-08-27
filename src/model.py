import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


# CNN Architecture

class AvianNetModelV1(nn.Module):

    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        # Input_shape = # of channels (3 in this case as its RGB)
        # Hidden_units is the in between shape? Like on that one website the number of squares
        # Output_shape is the number of classes (# of birds)
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
                      padding=1), # Maybe I should not pad because the Mel-Spectrograms don't really have much important data around the edges
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv_block_2 = nn.Sequential( #unclear if I wanna do another block rn
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

if __name__ == "__main__":
    torch.manual_seed(42)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    model_v1 = AvianNetModelV1(input_shape=3, 
                            hidden_units=10,
                            output_shape=5).to(device)


# Our data is [[batch_size],[channels],[height], [width]] --> [[32],[3],[128],[216]]



# image = torch.from_numpy(np.load(file=Path.cwd() / 'spec_augment_arr_tests' / 'test_arr.npy', allow_pickle=True)).unsqueeze(0).expand(3, -1, -1).unsqueeze(0).to(device)
# print(f"Test image original shape: {image.shape}")



# output = model_v1(image)
# print(output)



