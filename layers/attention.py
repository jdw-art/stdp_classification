import math
from typing import Tuple, Callable, List

import matplotlib.pyplot as plt
import torch
import torch.nn.functional

from layers.dense import DenseLayer, AttentionDenseLayer
from layers.embedding import EmbeddingLayer
from layers.encoding import EncodingLayer
from layers.reading import ReadingLayer, ReadingLayerReLU
from layers.writing import WritingLayer, WritingLayerReLU
from layers.memory import MemoryLayer
from models.neuron_models import NeuronModel
from models.protonet_models import SpikingProtoNet, ProtoNet
from policies import policy


class AttentionLayer(torch.nn.Module):
    def __init__(self, input_size: int, query_size: int, key_size: int, value_size: int,
                 memory_size: int, dynamics: NeuronModel):
        super().__init__()
        self.input_size = input_size
        self.query_size = query_size
        self.key_size = key_size
        self.value_size = value_size
        self.memory_size = memory_size
        self.dynamics = dynamics

        self.query_layer = AttentionDenseLayer(input_size, query_size, dynamics)
        self.key_layer = AttentionDenseLayer(memory_size, key_size, dynamics)
        self.value_layer = AttentionDenseLayer(memory_size, value_size, dynamics)
        self.attention_layer = AttentionDenseLayer(memory_size, memory_size, dynamics)
        # self.alpha = torch.nn.Parameter(torch.as_tensor(1.0))
        # self.sigmoid = torch.nn.Sigmoid()

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):

        query_output, _, _ = self.query_layer(query)
        key_output, _, _ = self.key_layer(key)
        value_output, _, _ = self.value_layer(value)

        attention = torch.bmm(query_output.transpose(1, 2), key_output)
        attention_value = torch.bmm(attention, value_output.transpose(1, 2)).transpose(1, 2)
        mem_output, _, _ = self.attention_layer(attention_value)

        # alpha = self.sigmoid(self.alpha)
        #
        # attention_output = (1 - alpha) * mem_output + alpha * value_output
        attention_output = mem_output

        return attention_output


class SpatioAttentionLayer(torch.nn.Module):
    def __init__(self, in_channel: int, out_channel: int, input_size: int, num_time_steps: int, dynamics: NeuronModel):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.input_size = input_size
        self.num_time_steps = num_time_steps
        self.conv2d = torch.nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=(num_time_steps, 1))
        self.linear = DenseLayer(self.input_size, self.input_size, dynamics)
        self.sigmoid = torch.nn.Sigmoid

    def forward(self, x: torch.Tensor):
        attention_x = x.unsqueeze(1)
        batch_size, _, _, _ = attention_x.size()
        factor = self.sigmoid(self.conv2d(attention_x).view(batch_size, -1)).unsqueeze(1)
        x, _, _ = self.linear(factor * x)
        return x
