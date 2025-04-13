import torch
from torch.cuda.amp import autocast
import numpy as np

def evaluate(model, test_data, batch_size=32, device="cuda", context_size=20):
    """评估固定上下文数量的误差"""
    model.eval()
    losses = []
    with torch.no_grad():
        for i in range(0, len(test_data), batch_size):
            batch = test_data[i:i + batch_size]
            X_batch = torch.stack([X[:context_size] for X, _, _, _ in batch])  # [batch, context_size, 20]
            Y_batch = torch.stack([Y[:context_size] for _, Y, _, _ in batch])  # [batch, context_size]
            x_query_batch = torch.stack([x_query for _, _, x_query, _ in batch])  # [batch, 20]
            y_query_batch = torch.tensor([[y_query] for _, _, _, y_query in batch], dtype=torch.float32).to(device)  # [batch, 1]
            
            with autocast():
                input_seq = torch.cat([X_batch, x_query_batch.unsqueeze(1)], dim=1).to(device)  # [batch, context_size+1, 20]
                output = model(input_seq, Y_batch)  # [batch, 1]
                loss = torch.nn.functional.mse_loss(output, y_query_batch)
            losses.append(loss.item() * len(batch))
        
        return np.sum(losses) / len(test_data)

def evaluate_with_varying_context(model, test_data, max_context=40, batch_size=32, device="cuda"):
    """评估不同上下文数量的误差"""
    errors = []
    for context_size in range(1, max_context + 1):
        mse = evaluate(model, test_data, batch_size, device, context_size)
        errors.append(mse)
    return errors