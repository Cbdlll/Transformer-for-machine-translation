import torch
import torch.nn as nn

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1, dim=-1, ignore_index=None):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred) # 创建一个与 pred 形状相同的零张量，用于存储平滑后的标签分布。
            true_dist.fill_(self.smoothing / (self.cls - 1)) # 平滑的核心步骤，将原本为 0 的概率值赋予一个小的概率值，smoothing/(k-1)。
            # 真实标签位置的概率值设置为 self.confidence（即 1 - smoothing）。
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
            
            
        loss = torch.sum(-true_dist * pred, dim=self.dim)

        # 如果设置了 ignore_index，则忽略填充符的损失
        if self.ignore_index is not None:
                ignore_mask = (target == self.ignore_index).unsqueeze(1).expand(-1, true_dist.size(1)) # 扩展mask，提高效率
                true_dist.masked_fill_(ignore_mask, 0.0) # 使用 masked_fill_ 更高效

        # 返回平均损失
        return torch.mean(loss)