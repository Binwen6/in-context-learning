import torch
import numpy as np

def flatten_prompt(X, Y, x_query):
    # 将 prompt 中的 (x_i, y_i) 拼接为交替序列，最后拼接 x_query
    # 每对 (x_i, y_i) 展平为 [d+1]，x_query 填充为 [d+1]
    xy_pairs = [torch.cat([x, y.unsqueeze(0)]) for x, y in zip(X, Y)]  # 每对: [d+1]
    x_query_padded = torch.cat([x_query, torch.tensor([0.0])])  # [d+1]
    sequence = torch.cat(xy_pairs + [x_query_padded])  # [(context_size+1)*(d+1)]
    return sequence

def generate_linear_data(num_prompts, context_size=20, d=20):
    prompts = []
    for _ in range(num_prompts):
        w = torch.randn(d)
        w = w / torch.norm(w)
        X = torch.randn(context_size, d)
        Y = X @ w + torch.randn(context_size) * 0.01
        x_query = torch.randn(d)
        y_query = x_query @ w
        flat_prompt = flatten_prompt(X, Y, x_query)
        prompts.append((flat_prompt, y_query, X, Y, x_query))
    return prompts

def generate_gaussian_kernel_data(num_prompts, context_size=20, d=20):
    prompts = []
    for _ in range(num_prompts):
        centers = torch.randn(5, d)
        X = torch.randn(context_size, d)
        Y = torch.stack([
            torch.exp(-torch.cdist(X, centers, p=2)**2 / 2).sum(dim=1)
        ]).squeeze(0)
        x_query = torch.randn(d)
        y_query = torch.exp(-torch.cdist(x_query.unsqueeze(0), centers, p=2)**2 / 2).sum()
        flat_prompt = flatten_prompt(X, Y, x_query)
        prompts.append((flat_prompt, y_query, X, Y, x_query))
    return prompts

def generate_nonlinear_dynamical_data(num_prompts, context_size=20, d=20):
    prompts = []
    for _ in range(num_prompts):
        x0 = torch.randn(d)
        X = [x0]
        for _ in range(context_size - 1):
            x_t = torch.tanh(X[-1] @ torch.randn(d, d)) + torch.randn(d) * 0.01
            X.append(x_t)
        X = torch.stack(X)
        Y = torch.sin(X @ torch.randn(d, 1)).squeeze()
        x_query = torch.tanh(X[-1] @ torch.randn(d, d)) + torch.randn(d) * 0.01
        y_query = torch.sin(x_query @ torch.randn(d))
        flat_prompt = flatten_prompt(X, Y, x_query)
        prompts.append((flat_prompt, y_query, X, Y, x_query))
    return prompts

# Generate datasets
train_linear = generate_linear_data(10000)
test_linear = generate_linear_data(1000)
train_gaussian = generate_gaussian_kernel_data(10000)
test_gaussian = generate_gaussian_kernel_data(1000)
train_dynamical = generate_nonlinear_dynamical_data(10000)
test_dynamical = generate_nonlinear_dynamical_data(1000)