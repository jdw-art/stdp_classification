"""Utilities for training and evaluation metrics"""

from typing import Tuple, Union, List

import torch


def accuracy(output: torch.Tensor, target: torch.Tensor, top_k: Tuple = (1,)) -> Union[float, List[float]]:
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        max_k = max(top_k)
        batch_size = target.size()[0]

        _, pred = output.topk(k=max_k, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in top_k:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))

        return res if len(top_k) > 1 else res[0]


def accuracy_qa(outputs: torch.Tensor, targets: torch.Tensor, top_k_list: List[int]):
    with torch.no_grad():
        batch_size, num_classes = targets.size()
        res = []
        batch_acc = 0.0

        for i in range(batch_size):
            output = outputs[i:i + 1, :]
            target = targets[i:i + 1, :]
            top_k = (top_k_list[i],)

            _, pred = output.topk(k=top_k[0], dim=1, largest=True, sorted=True)
            pred, _ = torch.sort(pred, dim=1, descending=False)

            # 将 one-hot 编码转换为标签索引
            target_idx = torch.nonzero(target.squeeze(), as_tuple=False).squeeze()

            # 调整 target_idx，使其与 pred 的维度匹配
            target_idx_expanded = target_idx.view(1, -1).expand_as(pred)

            # 仅当所有元素都是 True 时，才认为是正确分类
            correct = pred.eq(target_idx_expanded).all()

            if correct:
                batch_acc += 100.0 / batch_size

        res.append(batch_acc)
        return torch.tensor(res).to(outputs.device)

