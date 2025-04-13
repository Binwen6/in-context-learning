import torch
from mamba_ssm import Mamba2

class Mamba2ICL(torch.nn.Module):
    def __init__(self, d_model=20, d_state=64, d_conv=4, expand=2):
        super().__init__()
        self.mamba = Mamba2(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.output_head = torch.nn.Linear(d_model, 1)  # Predict scalar output

    def forward(self, x):
        x = self.mamba(x)
        return self.output_head(x)

