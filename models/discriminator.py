from typing import Tuple

import torch
import torch.nn as nn

from generator import getActivaiton

class discriminator(nn.Module):
    def __init__(
        self,
        imageChannel: int=3,
        layerChannels: Tuple[int]=(64, 128, 256, 512, 512, 1),
        activationType: str='relu'
    ):
        super().__init__()
        outChannel = imageChannel
        numLayer = len(layerChannels)
        
        self.convLayers = nn.ModuleList([])
        
        for i in range(numLayer):
            inChannel = outChannel
            outChannel = layerChannels[i]
            isFinal = (i == len(layerChannels) - 1)
            stride, padding = (1, 0) if isFinal else (2, 1)
            convLayer = getConvLayer(
                inChannel=inChannel,
                outChannel=outChannel,
                stride=stride,
                padding=padding,
                activationType=activationType,
                isFinal=isFinal
            )
            self.convLayers.append(convLayer)
    
    def forward(self, latent: torch.Tensor):
        for convLayer in self.convLayers:
            latent = convLayer(latent)
        return latent.view(-1)

def getConvLayer(
    inChannel,
    outChannel,
    kernelSize: Tuple[int]=(4, 4),
    stride: int=2,
    padding: int=1,
    activationType: str='relu',
    isFinal: bool=False
) -> nn.Module:
    conv = nn.Conv2d(inChannel, outChannel, kernelSize, stride, padding)
    batchNorm = nn.Identity() if isFinal else nn.BatchNorm2d(outChannel)
    activation = nn.Sigmoid() if isFinal else getActivaiton(activationType)
    
    layer = nn.Sequential(
        conv,
        batchNorm,
        activation
    )
    return layer

if __name__ == "__main__":
    image = torch.randn((10, 3, 128, 128))
    print("Input shape:")
    print(f"image: {image.shape}")
    
    model = discriminator()
    print(f"\nModel: {model}")
    
    output = model(image)
    print(f"\nOutput shape: {output.shape}")