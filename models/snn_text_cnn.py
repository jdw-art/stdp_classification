from collections import OrderedDict
from typing import List
import torch.nn as nn

import torch.nn.functional
from torch.nn import Parameter

from layers.convolution import *
from layers.dense import DenseLayer
from layers.encoding import EncodingLayer

from functions.autograd_functions import SpikeFunction
from models.neuron_models import NeuronModel, IafPscDelta, NonLeakyIafPscDelta
from snntorch import spikegen


# 脉冲卷积块：[卷积层，最大池化层]，使用IF神经元
def spiking_conv_block(in_channels: int, out_channels: int, use_bias: bool) -> torch.nn.Module:
    return torch.nn.Sequential(
        TextConv2DLayer(in_channels,
                        out_channels,
                        k_size=[3, 4, 5],
                        hidden_dim=300,
                        use_bias=use_bias,
                        dynamics=IafPscDelta(thr=0.03,
                                             perfect_reset=False,
                                             refractory_time_steps=3,
                                             spike_function=SpikeFunction,
                                             dampening_factor=1.0)),
        # torch.nn.BatchNorm2d(out_channels),
        # AvgPool2DLayer(k_size=2, stride=2, padding=0)
        TextAvgPool2DLayer(k_size=[3, 4, 5], sentence_length=1500)
    )


# SpikingCNN编码层
class SpikingTextCNN(torch.nn.Module):

    def __init__(self, dynamics: NeuronModel, num_time_steps: int = 10,
                 refractory_time_steps: int = 3, input_depth: int = 1,
                 hidden_size: int = 32, output_size: int = 100, use_bias: bool = False) -> None:
        super().__init__()
        self.input_depth = input_depth
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_time_steps = num_time_steps
        self.refractory_time_steps = refractory_time_steps
        self.dynamics = dynamics

        self.conv2d = TextConv2DLayer(input_depth,
                                      hidden_size,
                                      k_size=[3, 4, 5],
                                      hidden_dim=300,
                                      use_bias=use_bias,
                                      dynamics=IafPscDelta(thr=0.03,
                                                           perfect_reset=False,
                                                           refractory_time_steps=3,
                                                           spike_function=SpikeFunction,
                                                           dampening_factor=1.0))

        self.context_pool2d = TextAvgPool2DLayer(k_size=[3, 4, 5], sentence_length=1500)
        self.question_pool2d = TextAvgPool2DLayer(k_size=[3, 4, 5], sentence_length=200)

        self.linear = DenseLayer(self.hidden_size*3, output_size, self.dynamics)

    def forward(self, x: torch.Tensor, recall: bool = False) -> torch.Tensor:

        x = x.unsqueeze(1)
        sentence_encoded = []
        for i in range(x.size(2)):
            word_embedding = x[:, :, i:i+1, :]
            spike_data = spikegen.latency(word_embedding, num_steps=self.num_time_steps, tau=2, threshold=0.05,
                                          normalize=True, linear=True, clip=True)
            time_steps, batch_size, _, _, _ = spike_data.size()
            word_encoded = spike_data.permute(1, 0, 2, 3, 4).view(batch_size, time_steps, -1)
            sentence_encoded.append(word_encoded)

        sentence_encoded = torch.cat(sentence_encoded, dim=1)
        # sentence_encoded_array = sentence_encoded.clone().detach().to('cpu').numpy()
        x = self.conv2d(sentence_encoded)
        if recall:
            x = self.question_pool2d(x)
        else:
            x = self.context_pool2d(x)
        batch_size, time_steps, _, _, _ = x.size()
        x = x.reshape(batch_size, time_steps, -1)
        x, _, _ = self.linear(x)
        x_array = x.clone().detach().to('cpu').numpy()
        return x
