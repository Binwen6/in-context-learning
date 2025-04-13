# 导入标准库和第三方库
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.cuda.amp import autocast, GradScaler
import os

# 导入同级目录的自定义模块
from mamba2_icl import Mamba2ICL
from generate_icl_data import (generate_linear_data, generate_gaussian_kernel_data,
                              generate_nonlinear_dynamical_data)
from evaluate_icl import evaluate

# 设置随机种子以确保可复现性
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# 确认设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 创建输出目录用于保存中间文件
output_dir = "experiment_outputs"
os.makedirs(output_dir, exist_ok=True)