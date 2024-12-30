import math
from typing import Optional, Tuple, Callable, List

import torch
import torch.nn.functional
from torch.cuda import device

from functions.utility_functions import exp_convolve
from models.neuron_models import NeuronModel
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

## 自编码器+分类-三层
class STDPClassifyLayer(torch.nn.Module):

    def __init__(self, input_size: int, hidden_size: int, output_size: int, batch_size: int, plasticity_rule: Callable,
                 tau_trace: float, dynamics: NeuronModel) -> None:
        super().__init__()
        self.input_size = input_size  # 输入层的数据维度
        self.hidden_size = hidden_size  # 中间层的数据维度
        self.output_size = output_size  # 输出层维度
        self.batch_size = batch_size
        self.plasticity_rule = plasticity_rule  # 可塑性规则
        self.dynamics = dynamics

        self.decay_trace = math.exp(-1.0 / tau_trace)
        # 表示输入和l1层之间的权重
        self.input_weight = torch.nn.Parameter(torch.eye(input_size, input_size))
        # hidden层和input层权重
        self.hidden_weight = torch.nn.Parameter(torch.rand(self.hidden_size, self.input_size))
        # output层和hidden层权重
        self.output_weight = torch.nn.Parameter(torch.rand(self.output_size, self.hidden_size))

    def forward(self, x: torch.Tensor, train: bool, states: Optional[Tuple[
        List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]] = None) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        batch_size, sequence_length, _ = x.size()

        # hidden_weight_clone = self.hidden_weight.clone().detach().to('cpu').numpy()
        # output_weight_clone = self.output_weight.clone().detach().to('cpu').numpy()

        if states is None:
            l1_states = self.dynamics.initial_states(batch_size, self.input_size, x.dtype, x.device)
            l2_states = self.dynamics.initial_states(batch_size, self.hidden_size, x.dtype, x.device)
            l3_states = self.dynamics.initial_states(batch_size, self.output_size, x.dtype, x.device)
            l1_trace = torch.zeros(batch_size, self.input_size, dtype=x.dtype, device=x.device)
            l2_trace = torch.zeros(batch_size, self.hidden_size, dtype=x.dtype, device=x.device)
            l3_trace = torch.zeros(batch_size, self.output_size, dtype=x.dtype, device=x.device)
        else:
            l1_states, l2_states, l3_states, l1_trace, l2_trace, l3_trace = states

        # 编码后输入到key-value层的输入电流
        i = torch.nn.functional.linear(x, self.input_weight)
        # 每层神经元的输出
        l1_output_sequence = []
        l2_output_sequence = []
        l3_output_sequence = []

        # x_clone = x.clone().detach().to('cpu').numpy()
        # i_clone = i.clone().detach().to('cpu').numpy()

        # 执行自编码阶段
        if train:
            # training
            for t in range(sequence_length):
                # l1层神经元
                l1, l1_states = self.dynamics(i.select(1, t), l1_states)
                # l1层累计膜电位
                # l1v_t = 0.2 * (l1.unsqueeze(1) * self.hidden_weight).sum(2)
                l1v_t = 0.2 * (torch.nn.functional.linear(l1, self.hidden_weight))

                # l2层神经元
                l2, l2_states = self.dynamics(l1v_t, l2_states)

                # 更新l1层和l2层的迹
                l1_trace = exp_convolve(l1, l1_trace, self.decay_trace)
                l2_trace = exp_convolve(l2, l2_trace, self.decay_trace)
                # STDP
                delta_hidden_weight = self.plasticity_rule(l1_trace, l2_trace, self.hidden_weight.unsqueeze(0))
                self.hidden_weight.data = self.hidden_weight.data + delta_hidden_weight.squeeze(0)
                delta_hidden_weight_clone = delta_hidden_weight.clone().detach().to('cpu').numpy()


                # delta_hidden_weight = self.plasticity_rule(l1_trace, l2_trace, self.hidden_weight)
                # delta_hidden_weight_clone = delta_hidden_weight.clone().detach().to('cpu').numpy()
                # self.hidden_weight.data = self.hidden_weight.data + delta_hidden_weight

                # l2层累计膜电位
                # l2v_t = 0.2 * (l2.unsqueeze(1) * self.output_weight).sum(2)
                l2v_t = 0.2 * (torch.nn.functional.linear(l2, self.output_weight))

                l3, l3_states = self.dynamics(l2v_t, l3_states)

                # 计算l3层神经元的迹
                l3_trace = exp_convolve(l3, l3_trace, self.decay_trace)
                delta_output_weight = self.plasticity_rule(l2_trace, l3_trace, self.output_weight.unsqueeze(0))
                self.output_weight.data = self.output_weight.data + delta_output_weight.squeeze(0)
                delta_output_weight_clone = delta_output_weight.clone().detach().to('cpu').numpy()

                # delta_output_weight = self.plasticity_rule(l2_trace, l3_trace, self.output_weight)
                # delta_output_weight_clone = delta_output_weight.clone().detach().to('cpu').numpy()
                # self.output_weight.data = self.output_weight.data + delta_output_weight

                l1_output_sequence.append(l1)
                l2_output_sequence.append(l2)
                l3_output_sequence.append(l3)
        else:
            # testing
            for t in range(sequence_length):
                # l1层神经元
                l1, l1_states = self.dynamics(i.select(1, t), l1_states)
                # l1层累计膜电位
                l1v_t = 0.2 * (l1.unsqueeze(1) * self.hidden_weight).sum(2)

                # l2层神经元
                l2, l2_states = self.dynamics(l1v_t, l2_states)

                # l2层累计膜电位
                l2v_t = 0.2 * (l2.unsqueeze(1) * self.output_weight).sum(2)
                l3, l3_states = self.dynamics(l2v_t, l3_states)

                l1_output_sequence.append(l1)
                l2_output_sequence.append(l2)
                l3_output_sequence.append(l3)

        # hidden_weight_clone1 = self.hidden_weight.clone().detach().to('cpu').numpy()
        # output_weight_clone1 = self.output_weight.clone().detach().to('cpu').numpy()

        states = [l1_states, l2_states, l3_states, l1_trace, l2_trace, l3_trace]

        return torch.stack(l1_output_sequence, dim=1), \
               torch.stack(l2_output_sequence, dim=1), torch.stack(l3_output_sequence, dim=1), states

## 自编码器+分类-两层
class STDPClassifyLayerTwo(torch.nn.Module):

    def __init__(self, input_size: int, hidden_size: int, output_size: int, batch_size: int, plasticity_rule: Callable,
                 tau_trace: float, dynamics: NeuronModel) -> None:
        super().__init__()
        self.input_size = input_size  # 输入层的数据维度
        self.hidden_size = hidden_size  # 中间层的数据维度
        self.output_size = output_size  # 输出层维度
        self.batch_size = batch_size
        self.plasticity_rule = plasticity_rule  # 可塑性规则
        self.dynamics = dynamics

        self.decay_trace = math.exp(-1.0 / tau_trace)
        # 表示输入和l1层之间的权重
        self.input_weight = torch.rand(self.hidden_size, input_size)
        # hidden层和input层权重
        self.hidden_weight = torch.rand(self.output_size, self.hidden_size)

    def forward(self, x: torch.Tensor, train: bool, states: Optional[Tuple[
        List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]] = None) \
            -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        batch_size, sequence_length, _ = x.size()
        self.input_weight = self.input_weight.to(x.device)
        self.hidden_weight = self.hidden_weight.to(x.device)

        if states is None:
            l1_states = self.dynamics.initial_states(batch_size, self.hidden_size, x.dtype, x.device)
            l2_states = self.dynamics.initial_states(batch_size, self.output_size, x.dtype, x.device)
            l1_trace = torch.zeros(batch_size, self.hidden_size, dtype=x.dtype, device=x.device)
            l2_trace = torch.zeros(batch_size, self.output_size, dtype=x.dtype, device=x.device)
        else:
            l1_states, l2_states, l1_trace, l2_trace = states

        # 输入脉冲的迹
        input_trace = torch.zeros(batch_size, self.input_size, dtype=x.dtype, device=x.device)

        # 每层神经元的输出
        l1_output_sequence = []
        l2_output_sequence = []

        # 执行自编码阶段
        if train:
            # training
            for t in range(sequence_length):
                input_spk = x.select(1, t)
                input_trace = exp_convolve(input_spk, input_trace, self.decay_trace)

                # 输入电流
                input_current = 0.2 * (torch.nn.functional.linear(input_spk, self.input_weight))

                # l1层神经元
                l1, l1_states = self.dynamics(input_current, l1_states)
                l1_trace = exp_convolve(l1, l1_trace, self.decay_trace)

                # l1层累计膜电位
                l1v_t = 0.2 * (torch.nn.functional.linear(l1, self.hidden_weight))

                # l2层神经元
                l2, l2_states = self.dynamics(l1v_t, l2_states)
                l2_trace = exp_convolve(l2, l2_trace, self.decay_trace)

                # STDP
                # input-hidden
                delta_input_weight = self.plasticity_rule(input_trace, l1_trace, self.input_weight.unsqueeze(0)).detach()
                with torch.no_grad():
                    self.input_weight = self.input_weight + delta_input_weight.squeeze(0)

                # hidden-output
                delta_hidden_weight = self.plasticity_rule(l1_trace, l2_trace, self.hidden_weight.unsqueeze(0)).detach()
                with torch.no_grad():
                    self.hidden_weight = self.hidden_weight + delta_hidden_weight.squeeze(0)

                l1_output_sequence.append(l1)
                l2_output_sequence.append(l2)
        else:
            # testing
            for t in range(sequence_length):
                input_spk = x.select(1, t)
                input_trace = exp_convolve(input_spk, input_trace, self.decay_trace)

                # 输入电流
                input_current = 0.2 * (torch.nn.functional.linear(input_spk, self.input_weight))

                # l1层神经元
                l1, l1_states = self.dynamics(input_current, l1_states)
                # l1层累计膜电位
                l1v_t = 0.2 * (torch.nn.functional.linear(l1, self.hidden_weight))

                # l2层神经元
                l2, l2_states = self.dynamics(l1v_t, l2_states)

                l1_output_sequence.append(l1)
                l2_output_sequence.append(l2)

        states = [l1_states, l2_states, l1_trace, l2_trace]

        return torch.stack(l1_output_sequence, dim=1), \
               torch.stack(l2_output_sequence, dim=1), states


## 自编码器+分类-两层-兴奋+抑制神经元
class STDPClassifyLayerExcInh(torch.nn.Module):

    def __init__(self, input_size: int, output_size: int, batch_size: int, plasticity_rule: Callable,
                 tau_trace: float, feedback_delay: int, dynamics: NeuronModel) -> None:
        super().__init__()
        self.input_size = input_size  # 输入层的数据维度
        self.output_size = output_size  # 输出层维度
        self.batch_size = batch_size
        self.plasticity_rule = plasticity_rule  # 可塑性规则
        self.dynamics = dynamics
        self.feedback_delay = feedback_delay

        self.decay_trace = math.exp(-1.0 / tau_trace)
        # 表示脉冲序列和输入层input_layer之间的权重，输入脉冲序列和输入层神经元一对一连接
        self.input_weight = 0.005 * torch.eye(self.input_size, self.input_size, device="cuda:0")
        # hidden层和input层权重
        self.hidden_weight = torch.rand(self.output_size, self.input_size, device="cuda:0")
        # 兴奋层-抑制层权重, 兴奋神经元和抑制神经元一对一连接
        self.exc_inh_weight = torch.rand(self.output_size, self.output_size, device="cuda:0")
        # 抑制层-兴奋层权重，抑制神经元和除对应的兴奋神经元连接
        self.inh_exc_weight = torch.zeros(self.output_size, self.output_size, device="cuda:0")
        # 对焦矩阵掩码
        self.mask = torch.eye(self.exc_inh_weight.size(-1), device="cuda:0").bool()

    def forward(self, x: torch.Tensor, train: bool, states: Optional[Tuple[
        List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]] = None) \
            -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        batch_size, sequence_length, _ = x.size()

        if states is None:
            l1_states = self.dynamics.initial_states(batch_size, self.input_size, x.dtype, x.device)
            l2_states = self.dynamics.initial_states(batch_size, self.output_size, x.dtype, x.device)
            inhibitory_states = self.dynamics.initial_states(batch_size, self.output_size, x.dtype, x.device)
            l1_trace = torch.zeros(batch_size, self.input_size, dtype=x.dtype, device=x.device)
            l2_trace = torch.zeros(batch_size, self.output_size, dtype=x.dtype, device=x.device)
            inhibitory_trace = torch.zeros(batch_size, self.output_size, dtype=x.dtype, device=x.device)
            inhibitory_buffer = torch.zeros(batch_size, self.feedback_delay, self.output_size,
                                            dtype=x.dtype, device=x.device)
        else:
            l1_states, l2_states, inhibitory_states, l1_trace, l2_trace, inhibitory_trace, inhibitory_buffer = states

        # 编码后输入到key-value层的输入电流
        i = torch.nn.functional.linear(x, self.input_weight)
        # 每层神经元的输出
        l1_output_sequence = []
        l2_output_sequence = []

        # 执行自编码阶段
        if train:
            # training
            for t in range(sequence_length):
                with torch.no_grad():
                    # 保证兴奋-抑制神经元一对一连接
                    exc_inh_nondiag = self.exc_inh_weight.masked_fill(self.mask, 0.0)
                    self.exc_inh_weight = self.exc_inh_weight - exc_inh_nondiag
                    # 保证抑制神经元和除了对应的兴奋神经元之外的连接
                    self.inh_exc_weight = self.inh_exc_weight.masked_fill(self.mask, 0.0)

                # l1层神经元
                l1, l1_states = self.dynamics(i.select(1, t), l1_states)
                # l1层累计膜电位
                # l1v_t = 0.2 * (l1.unsqueeze(1) * self.hidden_weight).sum(2)
                l1v_t = (torch.nn.functional.linear(l1, self.hidden_weight))

                # 抑制层神经元到Response神经元的输入
                inhibitory_output = (inhibitory_buffer.select(1, t % self.feedback_delay).clone().unsqueeze(1)
                                     * self.inh_exc_weight.unsqueeze(0)).sum(2)

                # l2层神经元
                l2, l2_states = self.dynamics(l1v_t - 1.0 * inhibitory_output, l2_states)

                inhibitory_input = torch.nn.functional.linear(l2, self.exc_inh_weight)
                inhibitory, inhibitory_states = self.dynamics(inhibitory_input, inhibitory_states)
                # 更新缓冲区中用于存储过去的 inhibitory 层输出。这个缓冲区在每个时间步都会更新
                inhibitory_buffer[:, t % self.feedback_delay, :] = inhibitory.detach()

                # 更新l1层和l2层的迹
                l1_trace = exp_convolve(l1, l1_trace, self.decay_trace).detach()
                l2_trace = exp_convolve(l2, l2_trace, self.decay_trace).detach()
                # STDP
                delta_hidden_weight = self.plasticity_rule(l1_trace, l2_trace, self.hidden_weight.unsqueeze(0)).detach()
                with torch.no_grad():
                    self.hidden_weight += delta_hidden_weight.squeeze(0)

                # 更新兴奋-抑制层权重
                inhibitory_trace = exp_convolve(inhibitory, inhibitory_trace, self.decay_trace).detach()
                delta_exc_inh_weight = self.plasticity_rule(l2_trace, inhibitory_trace, self.exc_inh_weight.unsqueeze(0)).detach()
                delta_inh_exc_weight = self.plasticity_rule(inhibitory_trace, l2_trace, self.inh_exc_weight.unsqueeze(0)).detach()
                with torch.no_grad():
                    self.exc_inh_weight += delta_exc_inh_weight.squeeze(0)
                    self.inh_exc_weight += delta_inh_exc_weight.squeeze(0)

                l1_output_sequence.append(l1)
                l2_output_sequence.append(l2)
        else:
            # testing
            for t in range(sequence_length):
                # l1层神经元
                l1, l1_states = self.dynamics(i.select(1, t), l1_states)
                # l1层累计膜电位
                l1v_t = 0.2 * (l1.unsqueeze(1) * self.hidden_weight).sum(2)

                # 抑制层神经元到Response神经元的输入
                inhibitory_output = (inhibitory_buffer.select(1, t % self.feedback_delay).clone().unsqueeze(1)
                                     * self.inh_exc_weight.unsqueeze(0)).sum(2)

                # l2层神经元
                l2, l2_states = self.dynamics(l1v_t - 1.0 * inhibitory_output, l2_states)

                # 抑制层神经元更新
                inhibitory_input = torch.nn.functional.linear(l2, self.exc_inh_weight)
                inhibitory, inhibitory_states = self.dynamics(inhibitory_input, inhibitory_states)
                # 更新缓冲区中用于存储过去的 inhibitory 层输出。这个缓冲区在每个时间步都会更新
                inhibitory_buffer[:, t % self.feedback_delay, :] = inhibitory.detach()

                l1_output_sequence.append(l1)
                l2_output_sequence.append(l2)

        states = [l1_states, l2_states, inhibitory_states, l1_trace, l2_trace, inhibitory_trace, inhibitory_buffer]

        return torch.stack(l1_output_sequence, dim=1), \
            torch.stack(l2_output_sequence, dim=1), states