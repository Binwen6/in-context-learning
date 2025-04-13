import torch
from torch.cuda.amp import autocast
import numpy as np

def evaluate(model, test_data):
    model.eval()
    losses = []
    with torch.no_grad():
        for X, Y, x_query, y_query in test_data:
            with autocast():
                # 构造序列：20 个上下文 X + 1 个查询 x_query
                input_seq = torch.cat([X, x_query.unsqueeze(0)], dim=0).to("cuda")  # [21, 20]
                output = model(input_seq.unsqueeze(0))[:, -1, :]  # [1, 1]
                loss = torch.nn.functional.mse_loss(output, torch.tensor([y_query], dtype=torch.float32).to("cuda"))
            losses.append(loss.item())
    return np.mean(losses)