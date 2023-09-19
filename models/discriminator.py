from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor

from generator import getActivaiton

class Discriminator(nn.Module):
    def __init__(
        self,
        image_channel: int=3,
        layer_channels: Tuple[int]=(64, 128, 256, 512, 512, 1),
        activation_type: str='relu'
    ):
        super().__init__()
        out_channel = image_channel
        num_layer = len(layer_channels)
        
        self.conv_layers = nn.ModuleList([])
        
        for i in range(num_layer):
            in_channel = out_channel
            out_channel = layer_channels[i]
            is_final = (i == len(layer_channels) - 1)
            stride, padding = (1, 0) if is_final else (2, 1)
            conv_layer = getConvLayer(
                in_channel=in_channel,
                out_channel=out_channel,
                stride=stride,
                padding=padding,
                activation_type=activation_type,
                is_final=is_final
            )
            self.conv_layers.append(conv_layer)
    
    def forward(
        self,
        latent: Tensor
    ):
        for conv_layer in self.conv_layers:
            latent = conv_layer(latent)
        return latent.view(-1)

def getConvLayer(
    in_channel,
    out_channel,
    kernel_size: Tuple[int]=(4, 4),
    stride: int=2,
    padding: int=1,
    activation_type: str='relu',
    is_final: bool=False
) -> nn.Module:
    conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding)
    batch_norm = nn.Identity() if is_final else nn.BatchNorm2d(out_channel)
    activation = nn.Sigmoid() if is_final else getActivaiton(activation_type)
    
    layer = nn.Sequential(
        conv,
        batch_norm,
        activation
    )
    return layer

if __name__ == "__main__":
    image = torch.randn((10, 3, 128, 128))
    print("Input shape:")
    print(f"image: {image.shape}")
    
    model = Discriminator()
    print(f"\nModel: {model}")
    
    output = model(image)
    print(f"\nOutput shape: {output.shape}")
