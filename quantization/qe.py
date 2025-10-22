import copy
import torch
from torchao.quantization import Int4WeightOnlyConfig, quantize_



class ToyLinearModel(torch.nn.Module):
    def __init__(self, m: int, n: int, k: int):
        super().__init__()
        self.linear1 = torch.nn.Linear(m, n, bias=False)
        self.linear2 = torch.nn.Linear(n, k, bias=False)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x
    
model = ToyLinearModel(1024, 1024, 1024).eval().to(torch.bfloat16)


model = torch.compile(model, mode="max-autotune", fullgraph=True)
model_bf16 = copy.deepcopy(model)

quantize_(model, Int4WeightOnlyConfig(group_size=32))

print(model.linear1)
print(model.linear2)

