import torch.nn as nn
import torch

class ImageClassifier(nn.Module):
    """
    nn model inspired by TinyVGG to class object in a img
    """
    def __init__(self, input_layers: int, hidden_layers: int, output_layers: int):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(in_channels = input_layers,
                                            out_channels = hidden_layers,
                                            kernel_size = 3,
                                            stride = 1,
                                            padding = 0),
                                    # 126 x 126
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels = hidden_layers,
                                            out_channels = hidden_layers,
                                            kernel_size = 3,
                                            stride = 1,
                                            padding = 0),
                                    # 124 x 124
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size = 2)
                                    # 62 x 62
                                    )
        self.layer2 = nn.Sequential(nn.Conv2d(in_channels = hidden_layers,
                                            out_channels = hidden_layers,
                                            kernel_size = 3,
                                            stride = 1,
                                            padding = 0),
                                    # 60 x 60
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels = hidden_layers,
                                            out_channels = hidden_layers,
                                            kernel_size = 3,
                                            stride = 1,
                                            padding = 0),
                                    ## 58 x 58
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size = 2)
                                    # 29 x 29
                                    )
        self.layer3 = nn.Sequential(nn.Flatten(),
                                    nn.Linear(in_features = 29*29*hidden_layers,
                                            out_features = hidden_layers),
                                    nn.ReLU(),
                                    nn.Linear(in_features = hidden_layers,
                                            out_features = output_layers))
    def forward(self, x: torch.Tensor):
        return self.layer3(self.layer2(self.layer1(x)))

