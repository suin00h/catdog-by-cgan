from typing import Tuple

import torch
import torch.nn as nn

class generator(nn.Module):
    def __init__(
        self,
        noiseChannel: int,
        labelChannel: int,
        numLayer: int=5,
        layerChannels: tuple=(256, 128, 128, 64, 3),
        activationType: str='relu'
    ):
        super().__init__()
        outChannel = layerChannels[0]
        
        self.noiseLayer = getDeconvLayer(noiseChannel, outChannel, stride=1, padding=0)
        self.labelLayer = getDeconvLayer(labelChannel, outChannel, stride=1, padding=0)
        self.deconvLayers = nn.ModuleList([])
        
        outChannel = outChannel * 2
        for i in range(numLayer):
            inChannel = outChannel
            outChannel = layerChannels[i]
            isFinal = (i == numLayer - 1)
            deconvLayer = getDeconvLayer(
                inChannel=inChannel,
                outChannel=outChannel,
                activationType=activationType,
                isFinal=isFinal
            )
            self.deconvLayers.append(deconvLayer)
        
    def forward(
        self,
        noise,
        label
    ):
        noise = self.noiseLayer(noise)
        label = self.labelLayer(label)
        
        input = torch.cat((noise, label), dim=1)
        for deconvLayer in self.deconvLayers:
            input = deconvLayer(input)
        return input

def getDeconvLayer(
        inChannel,
        outChannel,
        kernelSize: Tuple[int]=(4, ),
        stride: int=2,
        padding: int=1,
        activationType: str='relu',
        isFinal: bool=False
    ) -> nn.Module:
    deconv = nn.ConvTranspose2d(inChannel, outChannel, kernelSize, stride, padding)
    batchNorm = nn.Identity() if isFinal else nn.BatchNorm2d(outChannel)
    activation = nn.Tanh() if isFinal else getActivaiton(activationType)
    
    layer = nn.Sequential(
        deconv,
        batchNorm,
        activation
    )
    return layer

def getActivaiton(activationType: str):
    if activationType == 'relu':
        relu = nn.ReLU()
    elif activationType == 'leakyrelu':
        relu = nn.LeakyReLU()
    elif activationType == 'gelu':
        relu = nn.GELU()
    else:
        print("Invalid activation type!")
        relu = nn.Identity()
    return relu