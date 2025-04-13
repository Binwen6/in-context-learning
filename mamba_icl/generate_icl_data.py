import torch
import numpy as np

def generate_linear_data(num_prompts, context_size=20, d=20):
    prompts = []
    for _ in range(num_prompts):
        w = torch.randn(d) / torch.norm(torch.randn(d))  # Normalized weights
        X = torch.randn(context_size, d)  # Inputs ~ N(0,1)
        Y = X @ w + torch.randn(context_size) * 0.01  # Linear fn + noise
        x_query = torch.randn(d)
        y_query = x_query @ w
        prompts.append((X, Y, x_query, y_query))
    return prompts

def generate_gaussian_kernel_data(num_prompts, context_size=20, d=20):
    prompts = []
    for _ in range(num_prompts):
        c = torch.randn(context_size, d)  # Random centers
        X = torch.randn(context_size, d)
        Y = torch.exp(-torch.cdist(X, c, p=2)**2 / 2)[:, 0]  # Gaussian kernel
        x_query = torch.randn(d)
        y_query = torch.exp(-torch.cdist(x_query.unsqueeze(0), c, p=2)**2 / 2)[0, 0]
        prompts.append((X, Y, x_query, y_query))
    return prompts

def generate_nonlinear_dynamical_data(num_prompts, context_size=20, d=20):
    prompts = []
    for _ in range(num_prompts):
        x0 = torch.randn(d)
        X = [x0]
        for _ in range(context_size-1):
            x_t = torch.tanh(X[-1]) + torch.randn(d) * 0.01  # Nonlinear update
            X.append(x_t)
        X = torch.stack(X)
        Y = torch.tanh(X)[:, 0]  # Output first dimension
        x_query = torch.tanh(X[-1]) + torch.randn(d) * 0.01
        y_query = torch.tanh(x_query)[0]
        prompts.append((X, Y, x_query, y_query))
    return prompts

# Generate datasets
train_linear = generate_linear_data(10000)
test_linear = generate_linear_data(1000)
train_gaussian = generate_gaussian_kernel_data(10000)
test_gaussian = generate_gaussian_kernel_data(1000)
train_dynamical = generate_nonlinear_dynamical_data(10000)
test_dynamical = generate_nonlinear_dynamical_data(1000)
