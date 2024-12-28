from collections import OrderedDict
from typing import List

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
        Conv2DLayer(in_channels,
                    out_channels,
                    k_size=3,
                    padding=1,
                    stride=1,
                    use_bias=use_bias,
                    dynamics=IafPscDelta(thr=0.05,
                                         perfect_reset=False,
                                         refractory_time_steps=3,
                                         spike_function=SpikeFunction,
                                         dampening_factor=1.0)),
        # torch.nn.BatchNorm2d(out_channels),
        # AvgPool2DLayer(k_size=2, stride=2, padding=0)
        MaxPool2DLayer(k_size=2, stride=2, padding=0)
    )


# SpikingCNN编码层
class SpikingProtoNet(torch.nn.Module):

    def __init__(self, dynamics: NeuronModel, weight_dict: dict = None, num_time_steps: int = 20,
                 refractory_time_steps: int = 3, input_depth: int = 1, input_size: int = 784,
                 hidden_size: int = 64, output_size: int = 64, use_bias: bool = False) -> None:
        super().__init__()
        self.input_depth = input_depth
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_time_steps = num_time_steps
        self.refractory_time_steps = refractory_time_steps
        self.dynamics = dynamics

        self.encoding = EncodingLayer(1, input_size, False, False, num_time_steps, dynamics)

        self.encoder = torch.nn.Sequential(
            spiking_conv_block(input_depth, hidden_size, use_bias),
            spiking_conv_block(hidden_size, hidden_size, use_bias),
            spiking_conv_block(hidden_size, hidden_size, use_bias),
            spiking_conv_block(hidden_size, output_size, use_bias),
        )

        if weight_dict:
            new_state_dict = OrderedDict()
            for k, v in weight_dict.items():
                if k.startswith('module.'):
                    k = k[len('module.'):]  # remove `module.`
                new_state_dict[k] = v
            for i in range(len(self.encoder)):
                self.encoder[i][0].conv2d.weight = Parameter(new_state_dict['encoder.' + str(i) + '.0.conv2d.weight'])
                # self.encoder[i][0].conv2d.weight.required_gard = False
            self.linear.W = Parameter(new_state_dict['linear1.W'])
            # self.linear.W.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # 将像素值输入神经元中进行编码
        # 需要将输入数据x在时间维度上复制，以模拟时间步，最后再加上一个全连接层dense层
        # [batch_size, channel, height, width]
        # x, _ = self.encoding(torch.flatten(x, -2, -1).unsqueeze(2)) # [batch_size, time_steps, height*width]
        # batch_size, time_steps, _ = x.size()
        # x = torch.reshape(x, [batch_size, self.num_time_steps, self.input_depth, 28, 28])

        # 不进行编码，直接重复输入像素值
        # batch_size, _, _, _ = x.size()
        # x = x.unsqueeze(1).expand(-1, self.num_time_steps, -1, -1, -1)

        # latency encoding延迟编码并且去除多余特征

        x = spikegen.latency(x, num_steps=self.num_time_steps, tau=2, threshold=0.05,
                             normalize=True, linear=True, clip=True)
        x = x.permute(1, 0, 2, 3, 4)
        batch_size, time_steps, _, _, _ = x.size()
        # spiking_array = x.view(batch_size, time_steps, -1).clone().detach().to('cpu').numpy()

        x = self.encoder(x)                     # [batch_size, time_steps, channel, height, width]
        return torch.reshape(x, [batch_size, time_steps, self.output_size])
