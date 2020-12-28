import numpy as np
import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torch.utils.data import TensorDataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
# y = 2 + 5x + epsilon(noise)
np.random.seed(7)
x = np.random.rand(100, 1)
y = 2 + 5 * x + .2 * np.random.randn(100, 1)

x_tensor = torch.from_numpy(x)
y_tensor = torch.from_numpy(y)

dataset = TensorDataset(x_tensor, y_tensor)
train_dataset, val_dataset = random_split(dataset, [80, 20])

train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=20)

class PyTorchLinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.randn(1, requires_grad=True))
        self.b = nn.Parameter(torch.randn(1, requires_grad=True))

    def forward(self, x):
        return self.a + self.b * x

torch.manual_seed(7)

model = PyTorchLinearRegression().to(device)
print(model.state_dict())

lr = 1e-1
epochs = 500

MSELoss = nn.MSELoss(reduction='mean')
optimizer = optim.SGD(model.parameters(), lr=lr)

def build_train_step(model, loss_fn, optimizer):
    def train_step(x, y):
        model.train()
        yhat = model(x)
        loss = loss_fn(y, yhat)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        return loss.item()
    return train_step

train_step = build_train_step(model, MSELoss, optimizer)

losses = []
val_losses = []
for epoch in range(epochs): 
    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        
        loss = train_step(x_batch, y_batch)
        losses.append(loss)

with torch.no_grad():
    for x, y in val_loader:
        x = x.to(device)
        y = y.to(device)
        model.eval()

        yhat = model(x)
        val_loss = MSELoss(y, yhat)
        val_losses.append(val_loss.item())

print(model.state_dict())
print(losses[-1])
print(val_losses)