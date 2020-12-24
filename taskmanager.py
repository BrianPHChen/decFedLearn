import numpy as np
import torch
from torch import nn
import io
import base64

class PyTorchLinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.randn(1, requires_grad=True))
        self.b = nn.Parameter(torch.randn(1, requires_grad=True))

    def forward(self, x):
        return self.a + self.b * x

model = PyTorchLinearRegression()
print(str(model.state_dict()))
# torch.save(model.state_dict(), 'model.pt')
buffer = io.BytesIO()
torch.save(model.state_dict(), buffer)
b64_data = base64.b64encode(buffer.getvalue())
print(b64_data.decode('utf-8'))