import numpy as np
import torch
from torch import nn
from torch import onnx
from torch.utils.data import TensorDataset
import base64
import io

# arr = np.loadtxt("data0.csv",delimiter=',')
# x, y = np.split(arr, 2, axis=1)

# x_tensor = torch.from_numpy(x)
# y_tensor = torch.from_numpy(y)

# dataset = TensorDataset(x_tensor, y_tensor)

class PyTorchLinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.randn(1, requires_grad=True))
        self.b = nn.Parameter(torch.randn(1, requires_grad=True))

    def forward(self, x):
        return self.a + self.b * x

data = 'UEsDBAAACAgAAAAAAAAAAAAAAAAAAAAAAAAQABIAYXJjaGl2ZS9kYXRhLnBrbEZCDgBaWlpaWlpaWlpaWlpaWoACY2NvbGxlY3Rpb25zCk9yZGVyZWREaWN0CnEAKVJxAShYAQAAAGFxAmN0b3JjaC5fdXRpbHMKX3JlYnVpbGRfdGVuc29yX3YyCnEDKChYBwAAAHN0b3JhZ2VxBGN0b3JjaApGbG9hdFN0b3JhZ2UKcQVYCAAAADM1OTk5MDcycQZYAwAAAGNwdXEHSwF0cQhRSwBLAYVxCUsBhXEKiWgAKVJxC3RxDFJxDVgBAAAAYnEOaAMoKGgEaAVYCAAAADY3ODQwMDE2cQ9oB0sBdHEQUUsASwGFcRFLAYVxEoloAClScRN0cRRScRV1fXEWWAkAAABfbWV0YWRhdGFxF2gAKVJxGFgAAAAAcRl9cRpYBwAAAHZlcnNpb25xG0sBc3NzYi5QSwcIK7kLYCQBAAAkAQAAUEsDBAAACAgAAAAAAAAAAAAAAAAAAAAAAAAVABkAYXJjaGl2ZS9kYXRhLzM1OTk5MDcyRkIVAFpaWlpaWlpaWlpaWlpaWlpaWlpaWuquiD1QSwcIeFBS4AQAAAAEAAAAUEsDBAAACAgAAAAAAAAAAAAAAAAAAAAAAAAVADkAYXJjaGl2ZS9kYXRhLzY3ODQwMDE2RkI1AFpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaEF1bPlBLBwhUrBeNBAAAAAQAAABQSwMEAAAICAAAAAAAAAAAAAAAAAAAAAAAAA8APwBhcmNoaXZlL3ZlcnNpb25GQjsAWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlozClBLBwjRnmdVAgAAAAIAAABQSwECAAAAAAgIAAAAAAAAK7kLYCQBAAAkAQAAEAAAAAAAAAAAAAAAAAAAAAAAYXJjaGl2ZS9kYXRhLnBrbFBLAQIAAAAACAgAAAAAAAB4UFLgBAAAAAQAAAAVAAAAAAAAAAAAAAAAAHQBAABhcmNoaXZlL2RhdGEvMzU5OTkwNzJQSwECAAAAAAgIAAAAAAAAVKwXjQQAAAAEAAAAFQAAAAAAAAAAAAAAAADUAQAAYXJjaGl2ZS9kYXRhLzY3ODQwMDE2UEsBAgAAAAAICAAAAAAAANGeZ1UCAAAAAgAAAA8AAAAAAAAAAAAAAAAAVAIAAGFyY2hpdmUvdmVyc2lvblBLBgYsAAAAAAAAAB4DLQAAAAAAAAAAAAQAAAAAAAAABAAAAAAAAAABAQAAAAAAANICAAAAAAAAUEsGBwAAAADTAwAAAAAAAAEAAABQSwUGAAAAAAQABAABAQAA0gIAAAAA'
data = base64.b64decode(data.encode('utf-8'))
buffer = io.BytesIO()
buffer.write(data)
buffer.seek(0)
model = PyTorchLinearRegression()
model.load_state_dict(torch.load(buffer))

print(model)
# print(model)
