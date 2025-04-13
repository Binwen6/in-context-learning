import torch
from torch.cuda.amp import autocast
import numpy as np

def evaluate(model, test_data, batch_size=32, device="cuda", context_size=20, d=20):
    """评估固定上下文数量的误差"""
    model.eval()
    losses = []
    with torch.no_grad():
        for i in range(0, len(test_data), batch_size):
            batch = test_data[i:i + batch_size]
            flat_prompt_batch = torch.stack([flat_prompt for flat_prompt, _, _, _, _ in batch])
            y_query_batch = torch.tensor([[y_query] for _, y_query, _, _, _ in batch], dtype=torch.float32).to(device)
            X_batch = torch.stack([X[:context_size] for _, _, X, _, _ in batch])
            Y_batch = torch.stack([Y[:context_size] for _, _, _, Y, _ in batch])
            
            with autocast():
                input_seq = flat_prompt_batch.view(-1, 2*context_size+1, d+1).to(device)
                output = model(input_seq, Y_batch)
                loss = torch.nn.functional.mse_loss(output, y_query_batch)
            losses.append(loss.item() * len(batch))
        
        return np.sum(losses) / len(test_data)

def evaluate_with_varying_context(model, test_data, max_context=40, batch_size=32, device="cuda"):
    """评估不同上下文数量的误差"""
    errors = []
    d = test_data[0][0].shape[-1] // (2*max_context + 1)  # 计算 d
    for context_size in range(1, max_context + 1):
        mse = evaluate(model, test_data, batch_size, device, context_size, d)
        errors.append(mse)
    return errors