import pytest
import torch
from ding.rl_utils import vtrace_data, vtrace_error
from ding.rl_utils.upgo import upgo_loss, upgo_returns, tb_cross_entropy


# @pytest.mark.unittest
# def test_upgo():
#     T, B, N, N2 = 4, 8, 5, 7
#
#     # # tb_cross_entropy: 3 tests
#     # logit = torch.randn(T, B, N, N2).softmax(-1).requires_grad_(True)
#     # action = logit.argmax(-1).detach()
#     # ce = tb_cross_entropy(logit, action)
#     # assert ce.shape == (T, B)
#     #
#     # logit = torch.randn(T, B, N, N2, 2).softmax(-1).requires_grad_(True)
#     # action = logit.argmax(-1).detach()
#     # with pytest.raises(AssertionError):
#     #     ce = tb_cross_entropy(logit, action)
#     #
#     # logit = torch.randn(T, B, N).softmax(-1).requires_grad_(True)
#     # action = logit.argmax(-1).detach()
#     # ce = tb_cross_entropy(logit, action)
#     # assert ce.shape == (T, B)
#
#     # upgo_returns
#
#     reward = torch.zeros(T, B)
#     reward[-1,:2] = 1.
#     reward[-1, 2:] = -1.
#     bootstrap_values = torch.zeros(T + 1, B).requires_grad_(True)
#     returns = upgo_returns(reward, bootstrap_values)
#     assert returns.shape == (T, B)
#
#     # upgo loss
#     logit = torch.zeros(T, B, N).softmax(-1).requires_grad_(True)
#     action = logit.argmax(-1).detach()
#     rhos = torch.ones(T, B)
#     loss = upgo_loss(logit, rhos, action, reward, bootstrap_values)
#     assert logit.requires_grad
#     assert bootstrap_values.requires_grad
#     for t in [logit, bootstrap_values]:
#         assert t.grad is None
#     loss.backward()
#     for t in [logit]:
#         assert isinstance(t.grad, torch.Tensor)
from torch.optim import SGD


@pytest.mark.unittest
def test_upgo2():
    S, B, A, N2 = 4, 8, 5, 7

    reward = torch.zeros(S, B)
    reward[-1,:2] = 1.
    reward[-1, 2:] = -1.
    bootstrap_values = torch.zeros(S + 1, B).requires_grad_(True)

    # upgo loss
    logit_q = torch.zeros((S, A),requires_grad=True)
    opt = SGD([logit_q],0.1)
    for i in range(10):
        logit = [logit_q.softmax(-1)] * B
        logit = torch.stack(logit)
        action = logit.argmax(-1).detach()
        rhos = torch.ones(S, B)
        loss = upgo_loss(logit, rhos, action, reward, bootstrap_values)
        assert logit.requires_grad
        assert bootstrap_values.requires_grad
        for t in [logit, bootstrap_values]:
            assert t.grad is None
        opt.zero_grad()
        loss.backward()
        opt.step()
# @pytest.mark.unittest
# def test_vtrace():
#     T, B, N = 4, 8, 16
#     value = torch.zeros(T + 1, B).requires_grad_(True)
#     reward = torch.zeros(T, B)
#     reward[-1,:2] = 1.
#     reward[-1, 2:] = -1.
#     target_output = torch.zeros(T, B, N).requires_grad_(True)
#     behaviour_output = torch.zeros(T, B, N)
#     action = torch.zeros(T, B) #torch.randint(0, N, size=(T, B))
#     data = vtrace_data(target_output, behaviour_output, action, value, reward, None)
#     loss = vtrace_error(data, rho_clip_ratio=1.1)
#     assert all([l.shape == tuple() for l in loss])
#     assert target_output.grad is None
#     assert value.grad is None
#     loss = sum(loss)
#     loss.backward()
#     assert isinstance(target_output, torch.Tensor)
#     assert isinstance(value, torch.Tensor)