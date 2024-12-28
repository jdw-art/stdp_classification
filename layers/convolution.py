"""Spiking 2D convolution and pooling layers"""

import math
import numpy as np
from typing import Tuple, Optional, List

import torch
import torch.nn.functional
from torch.nn.functional import conv2d, max_pool2d

from functions.autograd_functions import SpikeFunction
from models.neuron_models import NeuronModel, NonLeakyIafPscDelta


class Conv2DLayer(torch.nn.Module):

    def __init__(self, fan_in: int, fan_out: int, k_size: int, padding: int, stride: int,
                 dynamics: NeuronModel, use_bias: bool = False) -> None:
        super().__init__()
        self.fan_in = fan_in
        self.fan_out = fan_out
        self.k_size = k_size
        self.padding = padding
        self.stride = stride
        self.conv2d = torch.nn.Conv2d(fan_in, fan_out, (k_size, k_size), stride=(stride, stride),
                                      padding=(padding, padding), bias=use_bias)
        self.dynamics = dynamics
        self.reset_parameters()

    def forward(self, x: torch.Tensor, states: Optional[Tuple[torch.Tensor, ...]] = None) -> Tuple[torch.Tensor, ...]:
        batch_size, sequence_length, c, h, w = x.size()
        new_h = int((h - self.k_size + 2 * self.padding) / self.stride + 1)
        new_w = int((w - self.k_size + 2 * self.padding) / self.stride + 1)
        hidden_size = self.fan_out * new_h * new_w
        assert self.fan_in == c

        if states is None:
            states = self.dynamics.initial_states(batch_size, hidden_size, x.dtype, x.device)

        output_sequence, max_activation = [], [-float('inf')]
        for t in range(sequence_length):
            output = torch.flatten(self.conv2d(x.select(1, t)), -3, -1)
            max_activation.append(torch.max(output))
            output, states = self.dynamics(output, states)
            output_sequence.append(output)

        output = torch.reshape(torch.stack(output_sequence, dim=1),
                               [batch_size, sequence_length, self.fan_out, new_h, new_w])

        return output, max(max_activation)

    def reset_parameters(self) -> None:
        torch.nn.init.xavier_uniform_(self.conv2d.weight, gain=math.sqrt(2))


class MaxPool2DLayer(torch.nn.Module):

    def __init__(self, k_size: int, stride: int, padding: int) -> None:
        super().__init__()
        self.k_size = k_size
        self.stride = stride
        self.padding = padding
        self.max_pool = torch.nn.MaxPool2d(kernel_size=self.k_size, stride=self.stride, padding=self.padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x[0] if isinstance(x, Tuple) else x
        batch_size, sequence_length, c, h, w = x.size()

        output_sequence = []
        for t in range(sequence_length):
            output_sequence.append(self.max_pool(x.select(1, t)))

        return torch.stack(output_sequence, dim=1)


class AvgPool2DLayer(torch.nn.Module):

    def __init__(self, k_size: int, stride: int, padding: int) -> None:
        super().__init__()
        self.k_size = k_size
        self.stride = stride
        self.padding = padding
        self.avg_pool = torch.nn.AvgPool2d(kernel_size=self.k_size, stride=self.stride, padding=self.padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x[0] if isinstance(x, Tuple) else x
        batch_size, sequence_length, c, h, w = x.size()

        output_sequence = []
        for t in range(sequence_length):
            output_sequence.append(self.avg_pool(x.select(1, t)))

        return torch.stack(output_sequence, dim=1)


# class TextConv2DLayer(torch.nn.Module):
#
#     def __init__(self, fan_in: int, fan_out: int, k_size: List[int], hidden_dim: int,
#                  dynamics: NeuronModel, use_bias: bool = False) -> None:
#         super().__init__()
#         self.fan_in = fan_in
#         self.fan_out = fan_out
#         self.k_size = k_size
#         self.hidden_dim = hidden_dim
#         self.conv2d_0 = torch.nn.Conv2d(fan_in, fan_out, (k_size[0], hidden_dim), bias=use_bias)
#         self.conv2d_1 = torch.nn.Conv2d(fan_in, fan_out, (k_size[1], hidden_dim), bias=use_bias)
#         self.conv2d_2 = torch.nn.Conv2d(fan_in, fan_out, (k_size[2], hidden_dim), bias=use_bias)
#         self.dynamics = dynamics
#         self.reset_parameters()
#
#     def forward(self, x: torch.Tensor, states_0: Optional[Tuple[torch.Tensor, ...]] = None,
#                 states_1: Optional[Tuple[torch.Tensor, ...]] = None,
#                 states_2: Optional[Tuple[torch.Tensor, ...]] = None) -> Tuple[torch.Tensor, ...]:
#         x = x.unsqueeze(1)
#         batch_size, channel, sentence_length, embedding_length = x.size()
#         hidden_size_0 = self.fan_out * (sentence_length - self.k_size[0] + 1)
#         hidden_size_1 = self.fan_out * (sentence_length - self.k_size[1] + 1)
#         hidden_size_2 = self.fan_out * (sentence_length - self.k_size[2] + 1)
#         assert self.fan_in == channel
#
#         if states_0 is None:
#             states_0 = self.dynamics.initial_states(batch_size, hidden_size_0, x.dtype, x.device)
#         if states_1 is None:
#             states_1 = self.dynamics.initial_states(batch_size, hidden_size_1, x.dtype, x.device)
#         if states_2 is None:
#             states_2 = self.dynamics.initial_states(batch_size, hidden_size_2, x.dtype, x.device)
#
#         output_sequence_0 = []
#         for t in range(sentence_length):
#             outputs = torch.flatten(self.conv2d_0(x), -3, -1)
#             output, states_0 = self.dynamics(outputs, states_0)
#             output_sequence_0.append(output)
#         output_sequence_1 = []
#         for t in range(sentence_length):
#             outputs = torch.flatten(self.conv2d_1(x), -3, -1)
#             output, states_1 = self.dynamics(outputs, states_1)
#             output_sequence_1.append(output)
#         output_sequence_2 = []
#         for t in range(sentence_length):
#             outputs = torch.flatten(self.conv2d_2(x), -3, -1)
#             output, states_2 = self.dynamics(outputs, states_2)
#             output_sequence_2.append(output)
#
#         output_0 = torch.reshape(torch.stack(output_sequence_0, dim=1), [batch_size, sentence_length, self.fan_out,
#                                                                          sentence_length - self.k_size[0] + 1, 1])
#         output_1 = torch.reshape(torch.stack(output_sequence_1, dim=1), [batch_size, sentence_length, self.fan_out,
#                                                                          sentence_length - self.k_size[1] + 1, 1])
#         output_2 = torch.reshape(torch.stack(output_sequence_2, dim=1), [batch_size, sentence_length, self.fan_out,
#                                                                          sentence_length - self.k_size[2] + 1, 1])
#         return output_0, output_1, output_2
#
#     def reset_parameters(self) -> None:
#         torch.nn.init.xavier_uniform_(self.conv2d_0.weight, gain=math.sqrt(2))
#         torch.nn.init.xavier_uniform_(self.conv2d_1.weight, gain=math.sqrt(2))
#         torch.nn.init.xavier_uniform_(self.conv2d_2.weight, gain=math.sqrt(2))
#
#
# class TextMaxPool2DLayer(torch.nn.Module):
#
#     def __init__(self, k_size: List[int], sentence_length: int) -> None:
#         super().__init__()
#         self.k_size = k_size
#         self.sentence_length = sentence_length
#         self.max_pool_0 = torch.nn.MaxPool2d((sentence_length - k_size[0] + 1, 1))
#         self.max_pool_1 = torch.nn.MaxPool2d((sentence_length - k_size[1] + 1, 1))
#         self.max_pool_2 = torch.nn.MaxPool2d((sentence_length - k_size[2] + 1, 1))
#
#     def forward(self, x: Tuple[torch.Tensor, ...]) -> torch.Tensor:
#         x_0, x_1, x_2 = x
#         x_0 = x_0[0] if isinstance(x_0, Tuple) else x_0
#         batch_size, sequence_length, c, h, w = x_0.size()
#
#         output_sequence_0 = []
#         for t in range(sequence_length):
#             output_sequence_0.append(self.max_pool_0(x_0.select(1, t)))
#
#         x_1 = x_1[0] if isinstance(x_1, Tuple) else x_1
#         batch_size, sequence_length, c, h, w = x_1.size()
#
#         output_sequence_1 = []
#         for t in range(sequence_length):
#             output_sequence_1.append(self.max_pool_1(x_1.select(1, t)))
#
#         x_2 = x_2[0] if isinstance(x_2, Tuple) else x_2
#         batch_size, sequence_length, c, h, w = x_2.size()
#
#         output_sequence_2 = []
#         for t in range(sequence_length):
#             output_sequence_2.append(self.max_pool_2(x_2.select(1, t)))
#
#         output_0 = torch.stack(output_sequence_0, dim=1)
#         output_1 = torch.stack(output_sequence_1, dim=1)
#         output_2 = torch.stack(output_sequence_2, dim=1)
#
#         return torch.cat([output_0, output_1, output_2], dim=2)
#
#
# class TextAvgPool2DLayer(torch.nn.Module):
#
#     def __init__(self, k_size: List[int], sentence_length: int) -> None:
#         super().__init__()
#         self.k_size = k_size
#         self.avg_pool_0 = torch.nn.AvgPool2d((sentence_length - k_size[0] + 1, 1))
#         self.avg_pool_1 = torch.nn.AvgPool2d((sentence_length - k_size[1] + 1, 1))
#         self.avg_pool_2 = torch.nn.AvgPool2d((sentence_length - k_size[2] + 1, 1))
#
#     def forward(self, x: Tuple[torch.Tensor, ...]) -> torch.Tensor:
#         x_0, x_1, x_2 = x
#         x_0 = x_0[0] if isinstance(x_0, Tuple) else x_0
#         batch_size, sequence_length, c, h, w = x_0.size()
#
#         output_sequence_0 = []
#         for t in range(sequence_length):
#             output_sequence_0.append(self.avg_pool_0(x_0.select(1, t)))
#
#         x_1 = x_1[0] if isinstance(x_1, Tuple) else x_1
#         batch_size, sequence_length, c, h, w = x_1.size()
#
#         output_sequence_1 = []
#         for t in range(sequence_length):
#             output_sequence_1.append(self.avg_pool_1(x_1.select(1, t)))
#
#         x_2 = x_2[0] if isinstance(x_2, Tuple) else x_2
#         batch_size, sequence_length, c, h, w = x_2.size()
#
#         output_sequence_2 = []
#         for t in range(sequence_length):
#             output_sequence_2.append(self.avg_pool_2(x_2.select(1, t)))
#
#         output_0 = torch.stack(output_sequence_0, dim=1)
#         output_1 = torch.stack(output_sequence_1, dim=1)
#         output_2 = torch.stack(output_sequence_2, dim=1)
#
#         return torch.cat([output_0, output_1, output_2], dim=2)


class TextConv2DLayer(torch.nn.Module):
    def __init__(self, fan_in: int, fan_out: int, k_size: List[int], hidden_dim: int,
                 dynamics: NeuronModel, use_bias: bool = False) -> None:
        super().__init__()
        self.fan_in = fan_in
        self.fan_out = fan_out
        self.k_size = k_size
        self.hidden_dim = hidden_dim

        self.conv2d_0 = torch.nn.Conv2d(fan_in, fan_out, (k_size[0], hidden_dim), bias=use_bias)
        self.conv2d_1 = torch.nn.Conv2d(fan_in, fan_out, (k_size[1], hidden_dim), bias=use_bias)
        self.conv2d_2 = torch.nn.Conv2d(fan_in, fan_out, (k_size[2], hidden_dim), bias=use_bias)

        self.dynamics = dynamics
        self.reset_parameters()

    def forward(self, x: torch.Tensor, states: Optional[Tuple[torch.Tensor, ...]] = None) -> Tuple[torch.Tensor, ...]:
        x = x.unsqueeze(1)
        batch_size, channel, sentence_length, embedding_length = x.size()
        assert self.fan_in == channel

        hidden_sizes = [sentence_length - k + 1 for k in self.k_size]

        if states is None:
            states = [self.dynamics.initial_states(batch_size, self.fan_out * hidden_size, x.dtype, x.device)
                      for hidden_size in hidden_sizes]

        conv2d_layers = [self.conv2d_0, self.conv2d_1, self.conv2d_2]

        output_sequences = []

        for i, conv2d_layer in enumerate(conv2d_layers):
            hidden_size = hidden_sizes[i]
            states_i = states[i]
            output_sequence = []

            for t in range(sentence_length):
                outputs = torch.flatten(conv2d_layer(x), -3, -1)
                output, states_i = self.dynamics(outputs, states_i)
                output_sequence.append(output)

            output_i = torch.reshape(torch.stack(output_sequence, dim=1),
                                     [batch_size, sentence_length, self.fan_out, hidden_size, 1])
            output_sequences.append(output_i)

        return tuple(output_sequences)

    def reset_parameters(self) -> None:
        for conv2d_layer in [self.conv2d_0, self.conv2d_1, self.conv2d_2]:
            torch.nn.init.xavier_uniform_(conv2d_layer.weight, gain=math.sqrt(2))


class TextAvgPool2DLayer(torch.nn.Module):

    def __init__(self, k_size: List[int], sentence_length: int) -> None:
        super().__init__()
        self.k_size = k_size
        self.avg_pool_0 = torch.nn.AvgPool2d((sentence_length - k_size[0] + 1, 1))
        self.avg_pool_1 = torch.nn.AvgPool2d((sentence_length - k_size[1] + 1, 1))
        self.avg_pool_2 = torch.nn.AvgPool2d((sentence_length - k_size[2] + 1, 1))

    def forward(self, x: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        x_0, x_1, x_2 = x

        def apply_avg_pool(avg_pool, x):
            batch_size, sequence_length, c, h, w = x.size()
            output_sequence = [avg_pool(x.select(1, t)) for t in range(sequence_length)]
            return torch.stack(output_sequence, dim=1)

        output_0 = apply_avg_pool(self.avg_pool_0, x_0[0] if isinstance(x_0, Tuple) else x_0)
        output_1 = apply_avg_pool(self.avg_pool_1, x_1[0] if isinstance(x_1, Tuple) else x_1)
        output_2 = apply_avg_pool(self.avg_pool_2, x_2[0] if isinstance(x_2, Tuple) else x_2)

        return torch.cat([output_0, output_1, output_2], dim=2)

class SpikingConv:
    """
    Convolutional layer with IF spiking neurons that can fire only once.
    Implements a Winner-take-all STDP learning rule.
    """
    def __init__(self, input_shape, out_channels, kernel_size, stride, padding=0,
                nb_winners=1, firing_threshold=1, stdp_max_iter=None, adaptive_lr=False,
                stdp_a_plus=0.004, stdp_a_minus=-0.003, stdp_a_max=0.15, inhibition_radius=0,
                update_lr_cnt=500, weight_init_mean=0.8, weight_init_std=0.05, v_reset=0
        ):
        in_channels, in_height, in_width = input_shape
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride,stride)
        self.padding = padding if isinstance(padding, tuple) else (padding,padding)
        self.firing_threshold = firing_threshold
        self.v_reset = v_reset
        # 初始化权重，loc表示正态分布的均值，scale表示正态分布的标准差
        self.weights = np.random.normal(
            loc=weight_init_mean, scale=weight_init_std,
            size=(out_channels, in_channels, self.kernel_size[0], self.kernel_size[1]))

        # Output neurons
        out_height = int(((in_height + 2 * self.padding[0] - self.kernel_size[0]) / self.stride[0]) + 1)
        out_width = int(((in_width + 2 * self.padding[1] - self.kernel_size[1]) / self.stride[1]) + 1)
        self.pot = np.zeros((out_channels, out_height, out_width))
        # 用于标记活跃神经元
        self.active_neurons = np.ones(self.pot.shape).astype(bool)
        self.output_shape = self.pot.shape

        # STDP
        # 记录输入特征图的脉冲信号
        self.recorded_spks = np.zeros((in_channels, in_height+2*self.padding[0], in_width+2*self.padding[1]))
        self.nb_winners = nb_winners
        self.inhibition_radius = inhibition_radius
        self.adaptive_lr = adaptive_lr
        self.a_plus = stdp_a_plus
        self.a_minus = stdp_a_minus
        self.a_max = stdp_a_max
        # 记录stdp被调用的次数
        self.stdp_cnt = 0
        # 记录累积一定stdp学习后更新学习率次数
        self.update_lr_cnt = update_lr_cnt
        # stdp的最大迭代次数
        self.stdp_max_iter = stdp_max_iter
        self.plasticity = True
        self.stdp_neurons = np.ones(self.pot.shape).astype(bool)

    # 评估学习的收敛程度
    def get_learning_convergence(self):
        return (self.weights * (1-self.weights)).sum() / np.prod(self.weights.shape)

    def reset(self):
        # [：]进行广播赋值
        self.pot[:] = self.v_reset
        self.active_neurons[:] = True
        self.stdp_neurons[:] = True
        self.recorded_spks[:] = 0

    # 获得赢者通吃的winner
    def get_winners(self):
        winners = []
        channels = np.arange(self.pot.shape[0])
        # Copy potentials and keep neurons that can do STDP 标记哪些神经元可以进行STDP学习
        pots_tmp = np.copy(self.pot) * self.stdp_neurons
        # Find at most nb_winners
        while len(winners) < self.nb_winners:
            # Find new winner 先取神经元膜电位最大的一个作为初始winner
            winner = np.argmax(pots_tmp) # 1D index
            # 将winner的信息转换为三维信息以确定其的位置(channel, height, width)
            winner = np.unravel_index(winner, pots_tmp.shape) # 3D index
            # Assert winner potential is higher than firing threshold
            # If not, stop the winner selection
            # 如果winner的膜电压小于发射阈值，则停止选择winner
            if pots_tmp[winner] <= self.firing_threshold:
                break
            # Add winner 否则将winner添加到winners数组中
            winners.append(winner)
            # Disable winner selection for neurons in neighborhood of other channels
            # 禁用获胜神经元周围的其他神经元（垂直范围和水平范围）
            # 抑制处于不同通道内相邻的神经元，将这些神经元的膜电位设置为静息电位
            pots_tmp[channels != winner[0],
                max(0,winner[1]-self.inhibition_radius):winner[1]+self.inhibition_radius+1,
                max(0,winner[2]-self.inhibition_radius):winner[2]+self.inhibition_radius+1
            ] = self.v_reset
            # Disable winner selection for neurons in same channel
            # 禁用相同通道的神经元
            pots_tmp[winner[0]] = self.v_reset
        return winners

    # 横向抑制，抑制具有较低膜电位的神经元，增强具有较高膜电位的神经元；
    # 增强具有较高膜电位的神经元，抑制除该神经元所处通道外的其他神经元
    def lateral_inhibition(self, spks):
        # Get index of spikes
        spks_c, spks_h, spks_w = np.where(spks)
        # Get associated potentials 根据神经元索引获取对应的膜电位
        spks_pot = np.array([self.pot[spks_c[i],spks_h[i],spks_w[i]] for i in range(len(spks_c))])
        # Sort index by potential in a descending order 根据膜电位大小对其进行降序排列
        spks_sorted_ind = np.argsort(spks_pot)[::-1]
        # Sequentially inhibit neurons in the neighborhood of other channels
        # Neurons with highest potential inhibit neurons with lowest one, even if both spike
        for ind in spks_sorted_ind:
            # Check that neuron has not been inhibated by another one
            if spks[spks_c[ind], spks_h[ind], spks_w[ind]] == 1:
                # Compute index 如果放前神经元发放脉冲则需要抑制其他通道内的神经元
                inhib_channels = np.arange(spks.shape[0]) != spks_c[ind]
                # Inhibit neurons 使得其他通道内的神经元不发放脉冲
                spks[inhib_channels, spks_h[ind], spks_w[ind]] = 0
                # 并将其膜电压设置为静息电位
                self.pot[inhib_channels, spks_h[ind], spks_w[ind]] = self.v_reset
                # 将其神经元设置为非活性状态
                self.active_neurons[inhib_channels, spks_h[ind], spks_w[ind]] = False
        return spks

    # 计算输入数据和特定输出神经元之间的卷积
    def get_conv_of(self, input, output_neuron):
        # Neuron index 获取输出神经元的索引
        n_c, n_h, n_w = output_neuron
        # Get the list of convolutions on input neurons to update output neurons
        # shape : (in_neuron_values, nb_convs) 加上batch维度
        input = torch.Tensor(input).unsqueeze(0) # batch axis
        convs = torch.nn.functional.unfold(input, kernel_size=self.kernel_size, stride=self.stride)[0].numpy()
        # Get the convolution for the spiking neuron
        conv_ind = (n_h * self.pot.shape[2]) + n_w # 2D to 1D index
        return convs[:, conv_ind]

    # 进行STDP学习
    def stdp(self, winner):
        if not self.stdp_neurons[winner]: exit(1)
        if not self.plasticity: return
        # Count call 计算调用STDP的次数
        self.stdp_cnt += 1
        # Winner 3D coordinates 获取神经元的通道数、高、宽
        winner_c, winner_h, winner_w = winner
        # Get convolution window used to compute output neuron potential
        # 获取用于计算输出神经元膜电位的卷积窗口，并将其展平为一维数组
        conv = self.get_conv_of(self.recorded_spks, winner).flatten()
        # Compute dW 计算权重的更新值
        w = self.weights[winner_c].flatten() * (1 - self.weights[winner_c]).flatten()
        w_plus = conv > 0 # Pre-then-post
        w_minus = conv == 0 # Post-then-pre (we assume that if no spike before, then after)
        # 更新winner权重
        dW = (w_plus * w * self.a_plus) + (w_minus * w * self.a_minus)
        self.weights[winner_c] += dW.reshape(self.weights[winner_c].shape)
        # Lateral inhibition between channels (local inter competition) 横向抑制不同通道的相邻神经元
        channels = np.arange(self.pot.shape[0])
        self.stdp_neurons[channels != winner_c,
            max(0,winner_h-self.inhibition_radius):winner_h+self.inhibition_radius+1,
            max(0,winner_w-self.inhibition_radius):winner_w+self.inhibition_radius+1
        ] = False
        # Lateral inhibition in the same channel (gobal intra competition) 抑制相同通道的其他神经元
        self.stdp_neurons[winner_c] = False
        # Adpative learning rate 自适应学习率
        if self.adaptive_lr and self.stdp_cnt % self.update_lr_cnt == 0:
            self.a_plus = min(2 * self.a_plus, self.a_max)
            self.a_minus = - 0.75 * self.a_plus
        # Stop STDP after X trains 检查是否达到最大STDP学习次数，若是，则停止可塑性学习
        if self.stdp_max_iter is not None and self.stdp_cnt > self.stdp_max_iter:
            self.plasticity = False


    def __call__(self, spk_in, train=False):
        # padding 对输入信号进行填充
        spk_in = np.pad(spk_in, ((0,), (self.padding[0],), (self.padding[1],)), mode='constant')
        # Keep records of spike input for STDP 记录脉冲信号便于进行STDP学习
        self.recorded_spks += spk_in
        # Output recorded spikes 创建一个与输出神经元形状相同的全零数组，用于存储输出脉冲信号
        spk_out = np.zeros(self.pot.shape)
        # Convert to torch tensors 在spk_in的维度前加上一个batch=1的维度使其成为4维张量
        x = torch.Tensor(spk_in).unsqueeze(0) # Add batch axis for torch conv2d
        weights = torch.Tensor(self.weights) # converts at the fly... (not so good)
        # Convolve (using torch as it is fast and easier, to be changed)
        out_conv = conv2d(x, weights, stride=self.stride).numpy()[0] # Converted to numpy
        # Update potentials 更新具有活跃状态的神经元的电位，根据卷积操作的结果进行累加
        self.pot[self.active_neurons] += out_conv[self.active_neurons]
        # Check for neurons that can spike 判断神经元是否超过阈值
        output_spikes = self.pot > self.firing_threshold
        if np.any(output_spikes):
            # Generate spikes 将达到阈值的神经元标记为发放脉冲
            spk_out[output_spikes] = 1
            # Lateral inhibition for neurons in neighborhood in other channels
            # Inhibit and disable neurons with lower potential that fire 对发放脉冲的神经元进行横向抑制
            spk_out = self.lateral_inhibition(spk_out)
            # STDP plasticity
            if train and self.plasticity:
                # Find winners (based on potential)
                winners = self.get_winners()
                # Apply STDP for each neuron winner
                for winner in winners:
                    self.stdp(winner)
            # Reset potentials and disable neurons that fire 将发放脉冲后的神经元设置为静息状态
            self.pot[spk_out == 1] = self.v_reset
            self.active_neurons[spk_out == 1] = False
        return spk_out

class SpikingPool:
    """
    Pooling layer with spiking neurons that can fire only once.
    """
    def __init__(self, input_shape, kernel_size, stride, padding=0):
        in_channels, in_height, in_width = input_shape
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride,stride)
        self.padding = padding if isinstance(padding, tuple) else (padding,padding)
        out_height = int(((in_height + 2 * self.padding[0] - self.kernel_size[0]) / self.stride[0]) + 1)
        out_width = int(((in_width + 2 * self.padding[1] - self.kernel_size[1]) / self.stride[1]) + 1)
        self.output_shape = (in_channels, out_height, out_width)
        # Keep track of active neurons because they can fire once
        self.active_neurons = np.ones(self.output_shape).astype(bool)

    # 将初始神经元的状态全都设置为活跃状态
    def reset(self):
        self.active_neurons[:] = True

    def __call__(self, in_spks):
        # padding
        in_spks = np.pad(in_spks, ((0,), (self.padding[0],), (self.padding[1],)), mode='constant')
        in_spks = torch.Tensor(in_spks).unsqueeze(0)
        # Max pooling (using torch as it is fast and easier, to be changed)
        out_spks = max_pool2d(in_spks, self.kernel_size, stride=self.stride).numpy()[0]
        # Keep spikes of active neurons
        out_spks = out_spks * self.active_neurons
        # Update active neurons as each pooling neuron can fire only once
        self.active_neurons[out_spks == 1] = False
        return out_spks