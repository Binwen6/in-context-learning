import torch
from mamba_ssm import Mamba

class Mamba1ICL(torch.nn.Module):
    def __init__(self, d_model=20, d_state=64, d_conv=4, expand=2):
        super().__init__()
        self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.output_head = torch.nn.Linear(d_model, 1)
        self.y_projection = torch.nn.Linear(1, d_model)

    def forward(self, x, y=None):
        # 使用模型参数的设备
        device = next(self.parameters()).device
        x = x.to(device)  # 确保 x 在正确设备
        if y is not None:
            y = y.to(device)  # 确保 y 在正确设备
            seq = []
            for i in range(x.shape[1]):  # 20 上下文
                seq.append(x[:, i, :])  # X_i: [batch, 20]
                if i < y.shape[1]:  # Y_i: [batch]
                    y_i = self.y_projection(y[:, i].unsqueeze(-1))  # [batch, 20]
                    seq.append(y_i)
            seq = torch.stack(seq, dim=1)  # [batch, 40, 20]
            output = self.mamba(seq)[:, -1, :]  # 最后一个 Y_i 的输出
        else:
            output = self.mamba(x)[:, -1, :]
        return self.output_head(output)