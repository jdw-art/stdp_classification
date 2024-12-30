"""Network models"""

import math
from typing import Tuple, Callable, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional
from torch.nn.parameter import Parameter

from layers.classification import LinearClassifyLayer
from layers.convolution import Conv2DLayer, SpikingConv, SpikingPool
from layers.dense import DenseLayer, AttentionDenseLayer
from layers.embedding import EmbeddingLayer
from layers.encoding import EncodingLayer
from layers.stdp_classify_layer import STDPClassifyLayerExcInh
from models.neuron_models import NeuronModel, IafPscDelta

# STDP分类
class STDPClassifyExcInhNet(torch.nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, readout_delay: int, batch_size: int, tau_trace: float,
                 image_embedding_layer: torch.nn.Module, plasticity_rule: Callable, dynamics: NeuronModel) -> None:
        super().__init__()

        self.image_embedding_layer = image_embedding_layer
        self.classify_layer = STDPClassifyLayerExcInh(input_size, output_size, batch_size, plasticity_rule, tau_trace, readout_delay, dynamics)

    def forward(self, image: torch.Tensor, train: bool) -> Tuple:

        image_encoded = self.image_embedding_layer(image)

        _, outputs, _ = self.classify_layer(image_encoded, train)

        return outputs