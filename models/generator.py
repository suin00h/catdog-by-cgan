from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor

class Generator(nn.Module):
    def __init__(
        self,
        noise_channel: int,
        label_channel: int,
        layer_channels: Tuple[int]=(256, 128, 128, 64, 3),
        activation_type: str='relu'
    ):
        super().__init__()
        out_channel = layer_channels[0]
        num_layer = len(layer_channels)
        
        self.noise_layer = getDeconvLayer(noise_channel, out_channel, stride=1, padding=0)
        self.label_layer = getDeconvLayer(label_channel, out_channel, stride=1, padding=0)
        self.deconv_layers = nn.ModuleList([])
        
        out_channel = out_channel * 2
        for i in range(num_layer):
            in_channel = out_channel
            out_channel = layer_channels[i]
            is_final = (i == num_layer - 1)
            decon_layer = getDeconvLayer(
                in_channel=in_channel,
                out_channel=out_channel,
                activation_type=activation_type,
                is_final=is_final
            )
            self.deconv_layers.append(decon_layer)
    
    def forward(
        self,
        noise: Tensor,
        label: Tensor
    ):
        noise = self.noise_layer(noise)
        label = self.label_layer(label)
        
        latent = torch.cat((noise, label), dim=1)
        for deconv_layer in self.deconv_layers:
            latent = deconv_layer(latent)
        return latent # [3, 128, 128] generated image.

def getDeconvLayer(
    in_channel,
    out_channel,
    kernel_size: Tuple[int]=(4, 4),
    stride: int=2,
    padding: int=1,
    activation_type: str='relu',
    isFinal: bool=False
) -> nn.Module:
    deconv = nn.ConvTranspose2d(in_channel, out_channel, kernel_size, stride, padding)
    batch_norm = nn.Identity() if isFinal else nn.BatchNorm2d(out_channel)
    activation = nn.Tanh() if isFinal else getActivaiton(activation_type)
    
    layer = nn.Sequential(
        deconv,
        batch_norm,
        activation
    )
    return layer

def getActivaiton(activation_type: str):
    if activation_type == 'relu':
        relu = nn.ReLU()
    elif activation_type == 'leakyrelu':
        relu = nn.LeakyReLU()
    elif activation_type == 'gelu':
        relu = nn.GELU()
    else:
        print("Invalid activation type!")
        relu = nn.Identity()
    return relu

if __name__ == "__main__":
    noise_channel = 100
    label_channel = 3
    noise = torch.randn((10, noise_channel, 1, 1))
    label = torch.randn((10, label_channel, 1, 1))
    print("Input shapes:")
    print(f"noise: {noise.shape}")
    print(f"label: {label.shape}")
    
    model = Generator(noise_channel, label_channel)
    print(f"\nModel: {model}")
    
    output = model(noise, label)
    print(f"\nOutput shape: {output.shape}")
