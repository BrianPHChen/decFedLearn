import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch import optim
import base64
import io
import json
import websocket
import requests

try:
    import thread
except ImportError:
    import _thread as thread
import time

arr = np.loadtxt("data0.csv",delimiter=',')
x, y = np.split(arr, 2, axis=1)
x_tensor = torch.from_numpy(x)
y_tensor = torch.from_numpy(y)
dataset = TensorDataset(x_tensor, y_tensor)
train_loader = DataLoader(dataset=dataset, batch_size=16, shuffle=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

trainingRound = 0
training = False

class PyTorchLinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.randn(1, requires_grad=True))
        self.b = nn.Parameter(torch.randn(1, requires_grad=True))

    def forward(self, x):
        return self.a + self.b * x

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

def loadModel(buffer):
    model = PyTorchLinearRegression().to(device)
    lr = 1e-1
    MSELoss = nn.MSELoss(reduction='mean')
    optimizer = optim.SGD(model.parameters(), lr=lr)
    model.load_state_dict(torch.load(buffer))
    return model, build_train_step(model, MSELoss, optimizer)

def on_message(ws, message):
    global trainingRound
    global training

    blockObj = json.loads(message)
    blockHeight = blockObj["result"]["data"]["value"]["block"]["header"]["height"]
    # print("block: ", blockHeight)
    url = "http://127.0.0.1:26657"

    # check round
    payload = {
        "method": "abci_query",
        "params": {"path": "round"},
        "jsonrpc": "2.0",
        "id": 1,
    }

    response = requests.post(url, json=payload).json()
    roundVal = base64.b64decode(response['result']['response']['value'])
    roundVal = roundVal.decode('utf-8')
    roundVal = int(roundVal)

    if training == True or roundVal == 0:
        return
    else :
        training = True

    trainingRound = roundVal + 1
    print("[Start Training] Round:", trainingRound)
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
    model, train_step = loadModel(buffer)
    
    losses = []
    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        
        loss = train_step(x_batch, y_batch)
        losses.append(loss)

    print(model.state_dict())
    print(losses[-1])

    buffer.seek(0)
    buffer.truncate(0)
    torch.save(model.state_dict(), buffer)
    b64_data = base64.b64encode(buffer.getvalue())
    model = b64_data.decode('utf-8')

    modelTx = {"cid": 0, "round": trainingRound, "model": model}
    modelTx = json.dumps(modelTx)
    modelB64Str = base64.b64encode(modelTx.encode('utf-8')).decode('utf-8')
    payload = {
        "method": "broadcast_tx_sync",
        "params": [modelB64Str],
        "jsonrpc": "2.0",
        "id": 1,
    }
    response = requests.post(url, json=payload).json()
    training = False


def on_error(ws, error):
    print(error)

def on_close(ws):
    print("### closed ###")

def on_open(ws):
    def run(*args):
        data = json.dumps({ "jsonrpc": "2.0", "method": "subscribe", "params": ["tm.event='NewBlock'"], "id": 1 })
        ws.send(data)

    thread.start_new_thread(run, ())

if __name__ == "__main__":
    # websocket.enableTrace(True)
    ws = websocket.WebSocketApp("ws://localhost:26657/websocket",
                              on_message = on_message,
                              on_error = on_error,
                              on_close = on_close)
    ws.on_open = on_open
    ws.run_forever()
