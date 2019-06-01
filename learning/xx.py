import torch as t
from torch import nn
from torch.autograd import Variable as V


class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        # nn.Module.__init__(self)
        super(Linear, self).__init__()
        self.w = nn.Parameter(t.randn(in_features, out_features))  # nn.Parameter是特殊Variable
        self.b = nn.Parameter(t.randn(out_features))

    def forward(self, x):
        x = x.mm(self.w)
        return x + self.b


layer = Linear(4, 3)
input = V(t.randn(2, 4))
output = layer(input)
print(output)

for name, Parameter in layer.named_parameters():
    print(name, Parameter)