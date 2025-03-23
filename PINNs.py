import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os

# 确保结果目录存在
os.makedirs('results', exist_ok=True)

# 定义神经网络模型
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=[20, 20, 5], activation='tanh'):
        super(NeuralNetwork, self).__init__()
        layers = []
        layers.append(nn.Linear(input_size, hidden_size[0]))  # 输入层

        # 添加激活函数层
        activation_layer = {
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh()
        }.get(activation, nn.Softplus())
        layers.append(activation_layer)

        # 添加隐藏层和激活函数
        for i in range(len(hidden_size) - 1):
            layers.append(nn.Linear(hidden_size[i], hidden_size[i + 1]))
            layers.append(activation_layer)

        layers.append(nn.Linear(hidden_size[-1], output_size))  # 输出层
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# 自适应 tanh 激活函数
class AdaptiveTanh(nn.Module):
    def __init__(self, in_features):
        super(AdaptiveTanh, self).__init__()
        self.alpha = nn.Parameter(torch.ones(in_features))

    def forward(self, x):
        return torch.tanh(self.alpha * x)

# 带有自适应激活函数的神经网络
class ACNeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=[20, 20, 5]):
        super(ACNeuralNetwork, self).__init__()
        layers = [nn.Linear(input_size, hidden_size[0]), AdaptiveTanh(hidden_size[0])]
        for i in range(1, len(hidden_size)):
            layers += [nn.Linear(hidden_size[i-1], hidden_size[i]), AdaptiveTanh(hidden_size[i])]
        layers.append(nn.Linear(hidden_size[-1], output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# 梯度计算函数
def auto_grad(u, x, order=1):
    grad = u
    for _ in range(order):
        grad = torch.autograd.grad(
            outputs=grad,
            inputs=x,
            grad_outputs=torch.ones_like(grad),
            retain_graph=True,
            create_graph=True
        )[0]
    return grad

# Xavier 初始化
def xavier_init(layer):
    if isinstance(layer, nn.Linear):
        nn.init.xavier_uniform_(layer.weight)
        nn.init.zeros_(layer.bias)

# PINNs 主类
class PINNs():
    def __init__(self, layers, activation='tanh', device='cpu', initial_lr=0.001, sadap=False, is_AC=False):
        self.device = device
        self.is_AC = is_AC
        
        if not is_AC:
            self.dnn = NeuralNetwork(layers[0], layers[-1], layers[1:-1], activation).to(device)
        else:
            self.dnn = ACNeuralNetwork(layers[0], layers[-1], layers[1:-1]).to(device)
        self.dnn.apply(xavier_init)

        # 优化器配置
        self.optimizer = torch.optim.LBFGS(
            self.dnn.parameters(),
            lr=initial_lr,
            max_iter=50000,
            history_size=50,
            line_search_fn="strong_wolfe"
        )
        self.sadap = sadap
        self.optimizer_Adam = torch.optim.Adam(self.dnn.parameters(), lr=initial_lr)
        if self.sadap:
            self.scheduler = lr_scheduler.StepLR(self.optimizer_Adam, step_size=1000, gamma=0.9)
        self.iter = 0

    def closure(self):
        self.optimizer.zero_grad()
        loss = self.lossf(self.dnn)
        loss.backward()
        if self.iter % 100 == 0:
            print(f'Iter {self.iter}: Loss = {loss.item():.6f}')
        self.iter += 1
        return loss

    def train(self, epochs, lossf, loss_title="default"):
        self.dnn.train()
        self.lossf = lossf
        loss_history = []
        
        # Adam 优化阶段
        for epoch in range(epochs):
            loss = lossf(self.dnn)
            self.optimizer_Adam.zero_grad()
            loss.backward()
            self.optimizer_Adam.step()
            if self.sadap:
                self.scheduler.step()
            loss_history.append(loss.item())
        
        # 保存训练曲线
        plt.figure(figsize=(10, 6))
        plt.plot(loss_history)
        plt.title(f'Training Loss ({loss_title})')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.savefig(f'results/loss_curve_{loss_title}.png')
        plt.close()
        np.save(f'results/loss_history_{loss_title}.npy', np.array(loss_history))

        # LBFGS 优化阶段
        self.optimizer.step(self.closure)

    def predict(self, X):
        self.dnn.eval()
        with torch.no_grad():
            return self.dnn(torch.tensor(X, dtype=torch.float32).to(self.device)).cpu().numpy()

# 配置参数
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Nf = 20000
N_b = 1000
x_min, x_max = -1, 1
y_min, y_max = -1, 1

# 边界条件和内部点生成函数
def boundary_generator(n, value, axis):
    points = torch.rand(n, 2, device=device) * (y_max - y_min) + y_min
    points[:, axis] = value
    return points.requires_grad_(True)

left = lambda n: (boundary_generator(n, x_min, 0), torch.zeros(n, 1, device=device))
right = lambda n: (boundary_generator(n, x_max, 0), torch.zeros(n, 1, device=device))
lower = lambda n: (boundary_generator(n, y_min, 1), torch.zeros(n, 1, device=device))
upper = lambda n: (boundary_generator(n, y_max, 1), torch.zeros(n, 1, device=device))

def interior(n):
    return (torch.rand(n, 2, device=device) * (x_max - x_min) + x_min).requires_grad_(True)

# PDE 方程定义
def pde_residual(model, points):
    x = points
    u = model(x)
    
    u_x = auto_grad(u, x, 1)[:, 0:1]
    u_y = auto_grad(u, x, 1)[:, 1:2]
    u_xx = auto_grad(u_x, x, 1)[:, 0:1]
    u_yy = auto_grad(u_y, x, 1)[:, 1:2]
    
    residual = u_xx + u_yy + u - torch.sin(np.pi*x[:,0:1])*torch.sin(np.pi*x[:,1:2])
    return residual

# 训练函数
def train_model(use_adaptive=False):
    layers = [2, 20, 20, 20, 20, 1]
    model = PINNs(layers=layers, activation='tanh', device=device, is_AC=use_adaptive)
    
    def loss_function(model):
        # 边界损失
        b_loss = 0
        for boundary in [left, right, lower, upper]:
            points, target = boundary(N_b)
            pred = model(points)
            b_loss += F.mse_loss(pred, target)
        
        # PDE 损失
        points = interior(Nf)
        residual = pde_residual(model, points)
        pde_loss = F.mse_loss(residual, torch.zeros_like(residual))
        
        return b_loss + pde_loss
    
    model.train(400, loss_function, loss_title='adaptive' if use_adaptive else 'standard')
    return model

# 训练并保存结果
standard_model = train_model(use_adaptive=False)
adaptive_model = train_model(use_adaptive=True)

# 结果可视化
def visualize_results(model, title):
    x = np.linspace(x_min, x_max, 100)
    y = np.linspace(y_min, y_max, 100)
    X, Y = np.meshgrid(x, y)
    grid = np.hstack((X.reshape(-1,1), Y.reshape(-1,1)))
    
    pred = model.predict(grid).reshape(100, 100)
    true = (grid[:,0] + grid[:,1]) * np.sin(np.pi*grid[:,0]) * np.sin(np.pi*grid[:,1])
    true = true.reshape(100, 100)
    
    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.imshow(pred, extent=[x_min, x_max, y_min, y_max], origin='lower')
    plt.title(f'Prediction ({title})')
    plt.colorbar()
    
    plt.subplot(132)
    plt.imshow(true, extent=[x_min, x_max, y_min, y_max], origin='lower')
    plt.title('True Solution')
    plt.colorbar()
    
    plt.subplot(133)
    plt.imshow(np.abs(pred - true), extent=[x_min, x_max, y_min, y_max], origin='lower')
    plt.title('Absolute Error')
    plt.colorbar()
    
    plt.savefig(f'results/results_{title}.png')
    plt.close()

visualize_results(standard_model, 'standard')
visualize_results(adaptive_model, 'adaptive')