import torch
from torch import nn
import requests
import json
import base64
import io

class PyTorchLinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.randn(1, requires_grad=True))
        self.b = nn.Parameter(torch.randn(1, requires_grad=True))

    def forward(self, x):
        return self.a + self.b * x

payload = {
    "method": "abci_query",
    "params": {"path": "round"},
    "jsonrpc": "2.0",
    "id": 1,
}

url = "http://localhost:26657"
response = requests.post(url, json=payload).json()
value = response["result"]["response"]["value"]
value = base64.b64decode(value.encode('utf-8'))
trainingRound = value.decode('utf-8')


payload = {
    "method": "abci_query",
    "params": {"path": "model"},
    "jsonrpc": "2.0",
    "id": 1,
}

response = requests.post(url, json=payload).json()
value = base64.b64decode(response['result']['response']['value'])

modelData = base64.b64decode(value)
buffer = io.BytesIO()
buffer.write(modelData)
buffer.seek(0)

model = PyTorchLinearRegression()
model.load_state_dict(torch.load(buffer))

print("round: {round}, model: {model}".format(round=trainingRound, model=model.state_dict()))
