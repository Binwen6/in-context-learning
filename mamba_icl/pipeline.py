import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from mamba_ssm import Mamba
import tqdm

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 超参数
d_model = 256  # 模型维度
n_layers = 12  # Mamba层数
d_input = 20   # 输入向量维度
context_length = 10  # 上下文示例数
batch_size = 64
num_tasks = 10000  # 训练任务数
lr = 0.0001
epochs = 100

# 数据生成函数：线性回归任务
def generate_linear_data(batch_size, d_input, context_length, device):
    """
    生成线性回归ICL数据
    返回：prompts [batch, (context_length*(d_input+1)+d_input)], targets [batch, 1]
    """
    # 随机生成权重 w
    w = torch.randn(batch_size, d_input, device=device)
    w = w / torch.norm(w, dim=-1, keepdim=True)  # 单位球面

    # 生成输入 x
    x = torch.randn(batch_size, context_length + 1, d_input, device=device)

    # 计算输出 y = w * x
    y = torch.einsum('bi,bji->bj', w, x)  # [batch, context_length+1]

    # 构造prompt：x_1, y_1, ..., x_k, y_k, x_{k+1}
    prompts = []
    for i in range(context_length):
        prompts.append(x[:, i])  # x_i
        prompts.append(y[:, i:i+1])  # y_i
    prompts.append(x[:, context_length])  # x_{k+1}
    prompts = torch.cat(prompts, dim=-1)  # [batch, (context_length*(d_input+1)+d_input)]

    # 验证prompt维度
    expected_prompt_dim = context_length * (d_input + 1) + d_input
    assert prompts.shape[-1] == expected_prompt_dim, \
        f"Prompt dim mismatch: expected {expected_prompt_dim}, got {prompts.shape[-1]}"

    # 目标：y_{k+1}
    targets = y[:, context_length:context_length+1]  # [batch, 1]

    return prompts, targets

# Mamba模型定义
class MambaICL(nn.Module):
    def __init__(self, d_input, context_length, d_model, n_layers):
        super().__init__()
        self.d_input = d_input
        self.context_length = context_length
        # 修正input_dim，反映实际prompt结构
        self.input_dim = context_length * (d_input + 1) + d_input
        
        # 输入嵌入层
        self.input_proj = nn.Linear(self.input_dim, d_model)
        
        # Mamba层
        self.mamba_layers = nn.ModuleList([
            Mamba(
                d_model=d_model,
                d_state=16,
                d_conv=4,
                expand=2
            ) for _ in range(n_layers)
        ])
        
        # 输出层
        self.output_proj = nn.Linear(d_model, 1)
        
    def forward(self, x):
        # x: [batch, (context_length*(d_input+1)+d_input)]
        # 验证输入维度
        assert x.shape[-1] == self.input_dim, \
            f"Input dim mismatch: expected {self.input_dim}, got {x.shape[-1]}"
        
        x = self.input_proj(x)  # [batch, d_model]
        
        # Mamba需要序列维度，添加伪序列维度
        x = x.unsqueeze(1)  # [batch, 1, d_model]
        for layer in self.mamba_layers:
            x = layer(x)  # [batch, 1, d_model]
        x = x.squeeze(1)  # [batch, d_model]
        
        # 输出预测
        out = self.output_proj(x)  # [batch, 1]
        return out

# 训练函数
def train(model, device, epochs, batch_size, d_input, context_length):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for _ in tqdm.tqdm(range(num_tasks // batch_size)):
            prompts, targets = generate_linear_data(batch_size, d_input, context_length, device)
            
            optimizer.zero_grad()
            outputs = model(prompts)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Avg Loss: {total_loss / (num_tasks // batch_size):.6f}")

# 测试函数
def test(model, device, batch_size, d_input, context_length):
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0
    num_test_tasks = 1000
    
    with torch.no_grad():
        for _ in range(num_test_tasks // batch_size):
            prompts, targets = generate_linear_data(batch_size, d_input, context_length, device)
            outputs = model(prompts)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    
    print(f"Test MSE: {total_loss / (num_test_tasks // batch_size):.6f}")

# 主函数
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 初始化模型
    model = MambaICL(d_input, context_length, d_model, n_layers).to(device)
    
    # 训练
    print("Starting training...")
    train(model, device, epochs, batch_size, d_input, context_length)
    
    # 测试
    print("Starting testing...")
    test(model, device, batch_size, d_input, context_length)

if __name__ == "__main__":
    main()