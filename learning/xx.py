import pytest
import torch
from ding.rl_utils.upgo import upgo_loss, upgo_returns, tb_cross_entropy
from torch.optim import SGD
import torch.nn.functional as F

def test_upgo2():
    S, B, A, N2 = 4, 8, 5, 7

    reward = torch.zeros(S, B,dtype=torch.float32)
    reward[-1,:2] = 1.
    reward[-1, 2:] = -1.
    bootstrap_values_orig = torch.zeros((S+1,),dtype=torch.float32).requires_grad_(True)

    # upgo loss
    logit_q = torch.zeros((S, A),dtype=torch.float32,requires_grad=True)
    opt = SGD([logit_q,bootstrap_values_orig],0.1)
    action = torch.zeros(S,B,dtype=torch.long)
    for i in range(10):
        logit = torch.stack([logit_q.softmax(-1)] * B).transpose(0,1)
        bootstrap_values = torch.stack([bootstrap_values_orig] * B).transpose(0,1)
        rhos = torch.ones(S, B)
        up_loss = upgo_loss(logit, rhos, action, reward, bootstrap_values)
        with torch.no_grad():
            returns = upgo_returns(reward, bootstrap_values)

        print("values:")
        print(bootstrap_values_orig.detach().numpy())
        mask = reward + bootstrap_values[1:] - bootstrap_values[:-1]
        print("mask:")
        print(mask.transpose(0,1).detach().numpy())
        print("returns:")
        print(returns.transpose(0,1).detach().numpy())
        v_loss = 1/2 *(returns - bootstrap_values[:-1]) ** 2
        v_loss = v_loss.mean()
        # assert logit.requires_grad
        # assert bootstrap_values.requires_grad
        # for t in [logit, bootstrap_values]:
        #     assert t.grad is None
        opt.zero_grad()
        loss = up_loss + v_loss
        loss.backward()
        opt.step()
def xx():
    tensor_0 = torch.arange(3, 12).view(3, 3)
    print(tensor_0)
    index = torch.tensor([2, 1, 0,2,1,1,2])
    tensor_1 = tensor_0.gather(0,index.unsqueeze(-1).expand((index.shape[0],tensor_0.shape[1])).long())
    print(tensor_1)
if __name__ == '__main__':
    xx()