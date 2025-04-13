# import torch
# import numpy as np

# def generate_linear_data(num_prompts, context_size=20, d=20):
#     prompts = []
#     for _ in range(num_prompts):
#         w = torch.randn(d) / torch.norm(torch.randn(d))  # Normalized weights
#         X = torch.randn(context_size, d)  # Inputs ~ N(0,1)
#         Y = X @ w + torch.randn(context_size) * 0.01  # Linear fn + noise
#         x_query = torch.randn(d)
#         y_query = x_query @ w
#         prompts.append((X, Y, x_query, y_query))
#     return prompts

# def generate_gaussian_kernel_data(num_prompts, context_size=20, d=20):
#     prompts = []
#     for _ in range(num_prompts):
#         c = torch.randn(context_size, d)  # Random centers
#         X = torch.randn(context_size, d)
#         Y = torch.exp(-torch.cdist(X, c, p=2)**2 / 2)[:, 0]  # Gaussian kernel
#         x_query = torch.randn(d)
#         y_query = torch.exp(-torch.cdist(x_query.unsqueeze(0), c, p=2)**2 / 2)[0, 0]
#         prompts.append((X, Y, x_query, y_query))
#     return prompts

# def generate_nonlinear_dynamical_data(num_prompts, context_size=20, d=20):
#     prompts = []
#     for _ in range(num_prompts):
#         x0 = torch.randn(d)
#         X = [x0]
#         for _ in range(context_size-1):
#             x_t = torch.tanh(X[-1]) + torch.randn(d) * 0.01  # Nonlinear update
#             X.append(x_t)
#         X = torch.stack(X)
#         Y = torch.tanh(X)[:, 0]  # Output first dimension
#         x_query = torch.tanh(X[-1]) + torch.randn(d) * 0.01
#         y_query = torch.tanh(x_query)[0]
#         prompts.append((X, Y, x_query, y_query))
#     return prompts

# # Generate datasets
# train_linear = generate_linear_data(10000)
# test_linear = generate_linear_data(1000)
# train_gaussian = generate_gaussian_kernel_data(10000)
# test_gaussian = generate_gaussian_kernel_data(1000)
# train_dynamical = generate_nonlinear_dynamical_data(10000)
# test_dynamical = generate_nonlinear_dynamical_data(1000)


import torch
import numpy as np

def flatten_prompt(X, Y, x_query):
    # 将 prompt 中的 (x_i, y_i) 拼接为一维序列，最后拼接 x_query
    xy_pairs = [torch.cat([x, y.unsqueeze(0)]) for x, y in zip(X, Y)]  # 每对: [d+1]
    sequence = torch.cat(xy_pairs + [x_query], dim=0)  # [context_size*(d+1) + d]
    return sequence

def generate_linear_data(num_prompts, context_size=20, d=20):
    prompts = []
    for _ in range(num_prompts):
        w = torch.randn(d)
        w = w / torch.norm(w)  # Normalized weights
        X = torch.randn(context_size, d)
        Y = X @ w + torch.randn(context_size) * 0.01
        x_query = torch.randn(d)
        y_query = x_query @ w
        flat_prompt = flatten_prompt(X, Y, x_query)
        prompts.append((flat_prompt, y_query))
    return prompts

def generate_gaussian_kernel_data(num_prompts, context_size=20, d=20):
    prompts = []
    for _ in range(num_prompts):
        centers = torch.randn(5, d)  # 5 Gaussian centers (增加复杂度)
        X = torch.randn(context_size, d)
        Y = torch.stack([
            torch.exp(-torch.cdist(X, centers, p=2)**2 / 2).sum(dim=1)
        ]).squeeze(0)  # 每点对所有中心求和 → [context_size]
        x_query = torch.randn(d)
        y_query = torch.exp(-torch.cdist(x_query.unsqueeze(0), centers, p=2)**2 / 2).sum()
        flat_prompt = flatten_prompt(X, Y, x_query)
        prompts.append((flat_prompt, y_query))
    return prompts

def generate_nonlinear_dynamical_data(num_prompts, context_size=20, d=20):
    prompts = []
    for _ in range(num_prompts):
        x0 = torch.randn(d)
        X = [x0]
        for _ in range(context_size - 1):
            x_t = torch.tanh(X[-1] @ torch.randn(d, d)) + torch.randn(d) * 0.01  # 更复杂的动力演化
            X.append(x_t)
        X = torch.stack(X)  # [context_size, d]
        Y = torch.sin(X @ torch.randn(d, 1)).squeeze()  # 多维非线性组合 → scalar 输出
        x_query = torch.tanh(X[-1] @ torch.randn(d, d)) + torch.randn(d) * 0.01
        y_query = torch.sin(x_query @ torch.randn(d))
        flat_prompt = flatten_prompt(X, Y, x_query)
        prompts.append((flat_prompt, y_query))
    return prompts

# Example usage:
train_linear = generate_linear_data(10000)
test_linear = generate_linear_data(1000)
train_gaussian = generate_gaussian_kernel_data(10000)
test_gaussian = generate_gaussian_kernel_data(1000)
train_dynamical = generate_nonlinear_dynamical_data(10000)
test_dynamical = generate_nonlinear_dynamical_data(1000)
