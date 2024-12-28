import math
from typing import Optional, Tuple, Callable, List

import torch
import torch.nn.functional

from functions.utility_functions import exp_convolve
from models.neuron_models import NeuronModel
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

## 自编码器+分类
class LinearClassifyLayer(torch.nn.Module):

    def __init__(self, input_size: int, hidden_size: int, output_size: int, batch_size: int, plasticity_rule: Callable, tau_trace: float,
                 feedback_delay: int, inhibition_size: int, dynamics: NeuronModel) -> None:
        super().__init__()
        self.input_size = input_size  # 输入层的数据维度
        self.hidden_size = hidden_size  # 中间层的数据维度
        self.output_size = output_size  # 输出层维度
        self.batch_size = batch_size
        self.plasticity_rule = plasticity_rule  # 可塑性规则
        self.feedback_delay = feedback_delay  # 反馈延迟
        self.inhibition_size = inhibition_size
        self.dynamics = dynamics

        self.decay_trace = math.exp(-1.0 / tau_trace)
        # 表示输入和l1层之间的权重
        self.W = torch.nn.Parameter(torch.eye(hidden_size, input_size))
        # 使用 register_buffer 定义 mem
        self.mem = torch.nn.Parameter(torch.rand(self.batch_size, self.output_size, self.hidden_size))

    def forward(self, x: torch.Tensor, train: bool, states: Optional[Tuple[
        List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]] = None) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        batch_size, sequence_length, _ = x.size()

        if states is None:
            l1_states = self.dynamics.initial_states(batch_size, self.hidden_size, x.dtype, x.device)
            l2_states = self.dynamics.initial_states(batch_size, self.output_size, x.dtype, x.device)
            l1_trace = torch.zeros(batch_size, self.hidden_size, dtype=x.dtype, device=x.device)
            l2_trace = torch.zeros(batch_size, self.output_size, dtype=x.dtype, device=x.device)
        else:
            l1_states, l2_states, l1_trace, l2_trace = states

        # 编码后输入到key-value层的输入电流
        i = torch.nn.functional.linear(x, self.W)
        i_clone = i.clone().detach().to('cpu').numpy()
        # key和value层的脉冲输出
        l1_output_sequence = []
        l2_output_sequence = []

        # 执行自编码阶段
        if train:
            # training
            for t in range(sequence_length):
                # l1层神经元
                l1, l1_states = self.dynamics(i.select(1, t), l1_states)
                l1_clone = l1.clone().detach().to('cpu').numpy()
                # l1层累计膜电位
                l1v_t = 0.2 * (l1.unsqueeze(1) * self.mem).sum(2)

                # l2层神经元
                l2, l2_states = self.dynamics(l1v_t, l2_states)
                l2_clone = l2.clone().detach().to('cpu').numpy()

                # 更新l1层和l2层的迹
                l1_trace = exp_convolve(l1, l1_trace, self.decay_trace)
                l2_trace = exp_convolve(l2, l2_trace, self.decay_trace)
                # 更新缓冲区中用于存储过去的 l2 层输出。这个缓冲区在每个时间步都会更新
                # STDP
                delta_mem = self.plasticity_rule(l1_trace, l2_trace, self.mem)
                mem_before = self.mem.clone().detach().to('cpu').numpy()
                self.mem.data = self.mem.data + delta_mem
                mem_after = self.mem.clone().detach().to('cpu').numpy()

                l1_output_sequence.append(l1)
                l2_output_sequence.append(l2)
        else:
            # testing
            for t in range(sequence_length):
                # l1层神经元
                l1, l1_states = self.dynamics(i.select(1, t), l1_states)
                # l1层累计膜电位
                l1v_t = 0.2 * (l1.unsqueeze(1) * self.mem).sum(2)

                # l2层神经元
                l2, l2_states = self.dynamics(l1v_t, l2_states)

                l1_output_sequence.append(l1)
                l2_output_sequence.append(l2)

        states = [l1_states, l2_states, l1_trace, l2_trace]

        return self.mem, torch.stack(l1_output_sequence, dim=1), \
               torch.stack(l2_output_sequence, dim=1), states

