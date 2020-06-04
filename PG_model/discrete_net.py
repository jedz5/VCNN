import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from tianshou.policy import PPOPolicy
from tianshou.data import Batch

class Net(nn.Module):
    def __init__(self, layer_num, state_shape, action_shape=0, device='cpu'):
        super().__init__()
        self.device = device
        self.inpipe = [
            nn.Linear(np.prod(state_shape), 2048),
            nn.ReLU(inplace=True)]
        for i in range(layer_num):
            self.inpipe += [nn.Linear(2048, 2048), nn.ReLU(inplace=True)]
        self.inpipe = nn.Sequential(*self.inpipe)
        self.inpipe.to(self.device)

    def forward(self, s):
        # if not isinstance(s, torch.Tensor):
        #     s = torch.tensor(s, device=self.device, dtype=torch.float)
        # batch = s.shape[0]
        # s = s.view(batch, -1)
        logits = self.inpipe(s)
        return logits


class Actor(nn.Module):
    def __init__(self, preprocess_net, action_shape):
        super().__init__()
        self.preprocess = preprocess_net
        self.last = nn.Linear(128, np.prod(action_shape))

    def forward(self, s, state=None, info={}):
        logits, h = self.preprocess(s, state)
        logits = F.softmax(self.last(logits), dim=-1)
        return logits, h


class Critic(nn.Module):
    def __init__(self, preprocess_net):
        super().__init__()
        self.preprocess = preprocess_net
        self.last = nn.Linear(128, 1)

    def forward(self, s):
        logits, h = self.preprocess(s, None)
        logits = self.last(logits)
        return logits
class H3_policy(PPOPolicy):
    def forward(self, batch, env, **kwargs):
        act_and_mask = self.actor(batch.obs, env=env, info=batch.info)

        return Batch(act_id=act_and_mask['act_id'],..., state=batch.obs, )

    def learn(self, batch, batch_size=None, repeat=1, **kwargs):
        self._batch = batch_size
        losses, clip_losses, vf_losses, ent_losses = [], [], [], []
        v = []
        old_log_prob = []
        with torch.no_grad():
            for b in batch.split(batch_size, permute=False):
                v.append(self.critic(b.obs))
                old_log_prob.append(self(b).dist.log_prob(
                    torch.tensor(b.act, device=v[0].device)))
        batch.v = torch.cat(v, dim=0)  # old value
        dev = batch.v.device
        batch.act = torch.tensor(batch.act, dtype=torch.float, device=dev)
        batch.logp_old = torch.cat(old_log_prob, dim=0)
        batch.returns = torch.tensor(
            batch.returns, dtype=torch.float, device=dev
        ).reshape(batch.v.shape)
        if self._rew_norm:
            mean, std = batch.returns.mean(), batch.returns.std()
            if std > self.__eps:
                batch.returns = (batch.returns - mean) / std
        batch.adv = batch.returns - batch.v
        if self._rew_norm:
            mean, std = batch.adv.mean(), batch.adv.std()
            if std > self.__eps:
                batch.adv = (batch.adv - mean) / std
        for _ in range(repeat):
            for b in batch.split(batch_size):
                dist = self(b).dist
                value = self.critic(b.obs)
                ratio = (dist.log_prob(b.act) - b.logp_old).exp().float()
                surr1 = ratio * b.adv
                surr2 = ratio.clamp(
                    1. - self._eps_clip, 1. + self._eps_clip) * b.adv
                if self._dual_clip:
                    clip_loss = -torch.max(torch.min(surr1, surr2),
                                           self._dual_clip * b.adv).mean()
                else:
                    clip_loss = -torch.min(surr1, surr2).mean()
                clip_losses.append(clip_loss.item())
                if self._value_clip:
                    v_clip = b.v + (value - b.v).clamp(
                        -self._eps_clip, self._eps_clip)
                    vf1 = (b.returns - value).pow(2)
                    vf2 = (b.returns - v_clip).pow(2)
                    vf_loss = .5 * torch.max(vf1, vf2).mean()
                else:
                    vf_loss = .5 * (b.returns - value).pow(2).mean()
                vf_losses.append(vf_loss.item())
                e_loss = dist.entropy().mean()
                ent_losses.append(e_loss.item())
                loss = clip_loss + self._w_vf * vf_loss - self._w_ent * e_loss
                losses.append(loss.item())
                self.optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(
                    self.actor.parameters()) + list(self.critic.parameters()),
                    self._max_grad_norm)
                self.optim.step()
        return {
            'loss': losses,
            'loss/clip': clip_losses,
            'loss/vf': vf_losses,
            'loss/ent': ent_losses,
        }
# class DQN(nn.Module):
#
#     def __init__(self, h, w, action_shape, device='cpu'):
#         super(DQN, self).__init__()
#         self.device = device
#
#         self.conv1 = nn.Conv2d(4, 16, kernel_size=5, stride=2)
#         self.bn1 = nn.BatchNorm2d(16)
#         self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
#         self.bn2 = nn.BatchNorm2d(32)
#         self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
#         self.bn3 = nn.BatchNorm2d(32)
#
#         def conv2d_size_out(size, kernel_size=5, stride=2):
#             return (size - (kernel_size - 1) - 1) // stride + 1
#
#         convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
#         convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
#         linear_input_size = convw * convh * 32
#         self.fc = nn.Linear(linear_input_size, 512)
#         self.head = nn.Linear(512, action_shape)
#
#     def forward(self, x, state=None, info={}):
#         if not isinstance(x, torch.Tensor):
#             x = torch.tensor(x, device=self.device, dtype=torch.float)
#         x = x.permute(0, 3, 1, 2)
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = F.relu(self.bn3(self.conv3(x)))
#         x = self.fc(x.reshape(x.size(0), -1))
#         return self.head(x), state
