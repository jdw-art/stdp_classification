"""Encoding layer"""

from typing import Optional, Tuple, List

import torch

from models.neuron_models import NeuronModel


# 将输入序列编码为一系列时间步上的脉冲序列
class EncodingLayer(torch.nn.Module):

    def __init__(self, input_size: int, hidden_size: int, mask_time_words: bool, learn_encoding: bool,
                 num_time_steps: int, dynamics: NeuronModel) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size          # 隐藏状态的维度，也是编码后的数据维度
        self.num_time_steps = num_time_steps
        self.dynamics = dynamics                # 神经元模型

        # 编码过程中，输入层和隐藏层的权重矩阵，可通过梯度下降修改
        self.encoding = torch.nn.Parameter(torch.Tensor(input_size, hidden_size), requires_grad=learn_encoding)

        if mask_time_words and learn_encoding:
            # Set gradient of time-words encoding to zero to avoid modifying them
            def mask_time_words_hook(grad):
                grad_modified = grad.clone()
                grad_modified[-1] = 0.0
                return grad_modified

            self.encoding.register_hook(mask_time_words_hook)

        self.reset_parameters()

    def forward(self, x: torch.Tensor, states: Optional[List[torch.Tensor]] = None) -> Tuple[torch.Tensor,
                                                                                             List[torch.Tensor]]:
        # 输入数据的形状：[batch_size, sequence_length, input_size, hidden_size]
        x_array = x.clone().detach().to('cpu').numpy()
        batch_size, sequence_length, _, _ = x.size()

        if states is None:
            states = self.dynamics.initial_states(batch_size, self.hidden_size, x.dtype, x.device)
        # 输入数据和权重矩阵加权，得到i，形状为(batch_size, sequence_length, hidden_size)
        i = torch.sum(self.encoding * x, dim=2)

        # 对每个输入序列，使用神经元模型计算其每个图像的像素点在每个时间步上的脉冲，最终输出的脉冲序列是包含每个像素点在100ms内的脉冲序列
        output_sequence = []
        for n in range(sequence_length):
            for t in range(self.num_time_steps):
                output, states = self.dynamics(i.select(1, n), states)

                output_sequence.append(output)
        # 返回所有序列生成的脉冲序列，形状为（batch_size, sequence_length*num_time_steps, hidden_size）
        return torch.stack(output_sequence, dim=1), states

    def reset_parameters(self) -> None:
        self.encoding.data.fill_(1.0)
