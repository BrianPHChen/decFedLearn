import numpy as np
import torch
from torch import nn
from torch import onnx
from torch.utils.data import TensorDataset

arr = np.loadtxt("data0.csv",delimiter=',')
x, y = np.split(arr, 2, axis=1)

x_tensor = torch.from_numpy(x)
y_tensor = torch.from_numpy(y)

dataset = TensorDataset(x_tensor, y_tensor)

class PyTorchLinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.randn(1, requires_grad=True))
        self.b = nn.Parameter(torch.randn(1, requires_grad=True))

    def forward(self, x):
        return self.a + self.b * x

model = PyTorchLinearRegression()
model.load_state_dict(torch.load('model.pt'))

onnx

print(model.state_dict())
print(model)
