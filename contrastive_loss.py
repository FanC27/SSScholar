import torch
import torch.nn as nn
import torch.nn.functional as F

class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, anchor, positive, negative):
        # 计算相似度
        # sim_pos = torch.exp(torch.cosine_similarity(anchor, positive, dim=-1) / self.temperature)
        # sim_neg = torch.exp(torch.cosine_similarity(anchor, negative, dim=-1) / self.temperature)

        sim_pos = F.cosine_similarity(anchor, positive, dim=1)
        sim_neg = F.cosine_similarity(anchor, negative, dim=1)

        # 计算损失
        numerator = sim_pos
        denominator = sim_pos + sim_neg
        loss = -torch.log(numerator / denominator).mean()

        return loss