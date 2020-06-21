
import pprint
import argparse
from torch.utils.tensorboard import SummaryWriter
import time
from PG_model.discrete_net import H3_net
import torch
import numpy as np
from torch import nn
from typing import Dict, List, Tuple, Union, Optional
from H3_battle import *
from tianshou.policy import PGPolicy
from tianshou.data import Batch, ReplayBuffer

import pygame
import H3_battleInterface
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='Pong')
    parser.add_argument('--seed', type=int, default=1626)
    parser.add_argument('--buffer-size', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--step-per-epoch', type=int, default=100)
    parser.add_argument('--collect-per-step', type=int, default=10)
    parser.add_argument('--repeat-per-collect', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--layer-num', type=int, default=1)
    parser.add_argument('--training-num', type=int, default=2)
    parser.add_argument('--test-num', type=int, default=2)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--render', type=float, default=0.)
    parser.add_argument(
        '--device', type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu')
    # ppo special
    parser.add_argument('--vf-coef', type=float, default=0.5)
    parser.add_argument('--ent-coef', type=float, default=0.0)
    parser.add_argument('--eps-clip', type=float, default=0.2)
    parser.add_argument('--max-grad-norm', type=float, default=0.5)
    parser.add_argument('--max_episode_steps', type=int, default=2000)
    args = parser.parse_known_args()[0]
    return args

dev = 'cuda'
def softmax(logits, mask_orig,dev,add_little = False):
    mask = torch.tensor(mask_orig,dtype=torch.float,device=dev)
    logits1 = torch.exp(logits)
    logits2 = logits1 * mask
    logits3 = logits2 / (torch.sum(logits2,dim=-1,keepdim=True) + 1E-9)
    if add_little:
        logits3 += 1E-9
    return logits3
class H3_policy(PGPolicy):
    r"""Implementation of Proximal Policy Optimization. arXiv:1707.06347

    :param torch.optim.Optimizer optim: the optimizer for actor and critic
        network.
    :param torch.distributions.Distribution dist_fn: for computing the action.
    :param float discount_factor: in [0, 1], defaults to 0.99.
    :param float max_grad_norm: clipping gradients in back propagation,
        defaults to ``None``.
    :param float eps_clip: :math:`\epsilon` in :math:`L_{CLIP}` in the original
        paper, defaults to 0.2.
    :param float vf_coef: weight for value loss, defaults to 0.5.
    :param float ent_coef: weight for entropy loss, defaults to 0.01.
    :param action_range: the action range (minimum, maximum).
    :type action_range: (float, float)
    :param float gae_lambda: in [0, 1], param for Generalized Advantage
        Estimation, defaults to 0.95.
    :param float dual_clip: a parameter c mentioned in arXiv:1912.09729 Equ. 5,
        where c > 1 is a constant indicating the lower bound,
        defaults to 5.0 (set ``None`` if you do not want to use it).
    :param bool value_clip: a parameter mentioned in arXiv:1811.02553 Sec. 4.1,
        defaults to ``True``.
    :param bool reward_normalization: normalize the returns to Normal(0, 1),
        defaults to ``True``.

    .. seealso::

        Please refer to :class:`~tianshou.policy.BasePolicy` for more detailed
        explanation.
    """
    def __init__(self,
                 net: torch.nn.Module,
                 optim: torch.optim.Optimizer,
                 dist_fn: torch.distributions.Distribution,
                 device,
                 discount_factor: float = 0.99,
                 max_grad_norm: Optional[float] = 0.5,
                 eps_clip: float = .2,
                 vf_coef: float = .5,
                 ent_coef: float = .01,
                 action_range: Optional[Tuple[float, float]] = None,
                 gae_lambda: float = 0.95,
                 dual_clip: float = 5.,
                 value_clip: bool = True,
                 reward_normalization: bool = False,
                 **kwargs) -> None:
        super().__init__(None, None, dist_fn, discount_factor, **kwargs)
        self._max_grad_norm = max_grad_norm
        self._eps_clip = eps_clip
        self._w_vf = vf_coef
        self._w_ent = ent_coef
        self._range = action_range
        self.ppo_net = net
        self.optim = optim
        self._batch = 64
        assert 0 <= gae_lambda <= 1, 'GAE lambda should be in [0, 1].'
        self._lambda = gae_lambda
        assert dual_clip is None or dual_clip > 1, \
            'Dual-clip PPO parameter should greater than 1.'
        self._dual_clip = dual_clip
        self._value_clip = value_clip
        self._rew_norm = reward_normalization
        self.__eps = np.finfo(np.float32).eps.item()
        self.dist_fn = dist_fn
        self.device = device
    def process_gae(self, batch: Batch) -> Batch:
        if self._rew_norm:
            mean, std = batch.rew.mean(), batch.rew.std()
            if std > self.__eps:
                batch.rew = (batch.rew - mean) / std
        if self._lambda in [0, 1]:
            return self.compute_episodic_return(
                batch, None, gamma=self._gamma, gae_lambda=self._lambda)
        v_ = []
        with torch.no_grad():
            for b in batch.split(self._batch, shuffle=False):
                v_.append(self.ppo_net(**b.obs_next,critic_only = True))
        v_ = torch.cat(v_, dim=0).cpu().numpy()
        return self.compute_episodic_return(
            batch, v_, gamma=self._gamma, gae_lambda=self._lambda)

    def forward(self, ind,attri_stack,planes_stack,plane_glb,
                env,print_act = False,
                **kwargs) -> Batch:
        act_logits, targets_logits, position_logits, spell_logits, value = self.ppo_net(ind,attri_stack,planes_stack,plane_glb)
        mask_acts = env.legal_act(level=0)
        #act_id = -1
        mask_position = np.zeros(11 * 17, dtype=bool)
        position_id = -1
        mask_targets = np.zeros(7, dtype=bool)
        target_id = -1
        mask_spell = np.zeros(10, dtype=bool)
        spell_id = -1
        act_logits = softmax(act_logits, mask_acts, self.device)
        if print_act:
            print(act_logits)
        act_id = self.dist_fn(act_logits).sample()[0].item()
        if act_id == actionType.move.value:
            mask_position = env.legal_act(level=1, act_id=act_id)
            position_logits = softmax(position_logits, mask_position, self.device)
            position_id = self.dist_fn(position_logits).sample()[0].item()
        elif act_id == actionType.attack.value:
            mask_targets = env.legal_act(level=1, act_id=act_id)
            targets_logits = softmax(targets_logits, mask_targets, self.device)
            if torch.sum(targets_logits,dim=-1,keepdim=True).item() > 0.5:
                target_id = self.dist_fn(targets_logits).sample()[0].item()
            else:
                logger.info("no attack target found!!")
                assert 0
            mask_position = env.legal_act(level=2, act_id=act_id, target_id=target_id)
            position_logits = softmax(position_logits, mask_position, self.device)
            if torch.sum(position_logits,dim=-1,keepdim=True).item() > 0.5:
                position_id = self.dist_fn(position_logits).sample()[0].item()
        return {'act_id': act_id, 'spell_id': spell_id, 'target_id': target_id, 'position_id': position_id,
                'mask_acts': mask_acts, 'mask_spell': mask_spell, 'mask_targets': mask_targets,
                'mask_position': mask_position, 'value': value}

    def learn(self, batch, batch_size=None, repeat=2, **kwargs):
        self._batch = batch_size
        losses, clip_losses, vf_losses, ent_losses = [], [], [], []
        v = []
        old_prob_act = []
        old_prob_position = []
        old_prob_target = []
        old_prob_spell = []
        with torch.no_grad():
            for b in batch.split(batch_size,shuffle=False):
                # obs = {'ind': ind, 'attri_stack': attri_stack, 'planes_stack': planes_stack, 'plane_glb': plane_glb}
                # acts = {'act_id': act_id, 'position_id': position_id, 'target_id': target_id, 'spell_id': spell_id}
                # mask = {'mask_acts': result['mask_acts'], 'mask_spell': result['mask_spell'],
                #         'mask_targets': result['mask_targets'], 'mask_position': result['mask_position']}
                mask_acts = b.info.mask_acts
                mask_spell = b.info.mask_spell
                mask_targets = b.info.mask_targets
                mask_position = b.info.mask_position

                # act_logits, targets_logits, position_logits, spell_logits, value = self.ppo_net(b.obs.ind,b.obs.attri_stack,b.obs.planes_stack,b.obs.plane_glb)
                act_logits, targets_logits, position_logits, spell_logits, value = self.ppo_net(**b.obs)
                v.append(value)

                act_logits = softmax(act_logits, mask_acts, self.device,add_little=True)
                old_prob_act.append(self.dist_fn(act_logits).log_prob(torch.tensor(b.act.act_id, device=self.device)))
                targets_logits = softmax(targets_logits, mask_targets, self.device,add_little=True)
                old_prob_target.append(self.dist_fn(targets_logits).log_prob(torch.tensor(b.act.target_id, device=self.device)))
                position_logits = softmax(position_logits, mask_position, self.device,add_little=True)
                old_prob_position.append(self.dist_fn(position_logits).log_prob(torch.tensor(b.act.position_id, device=self.device)))
                spell_logits = softmax(spell_logits, mask_spell, self.device,add_little=True)
                old_prob_spell.append(self.dist_fn(spell_logits).log_prob(torch.tensor(b.act.spell_id, device=self.device)))
        batch.v = torch.cat(v, dim=0)  # old value
        # batch.act.act_id = torch.tensor(batch.act.act_id, dtype=torch.float, device=self.device)
        # batch.act.target_id = torch.tensor(batch.act.target_id, dtype=torch.float, device=self.device)
        # batch.act.position_id = torch.tensor(batch.act.position_id, dtype=torch.float, device=self.device)
        # batch.act.spell_id = torch.tensor(batch.act.spell_id, dtype=torch.float, device=self.device)
        batch.old_prob_act = torch.cat(old_prob_act, dim=0)
        batch.old_prob_target = torch.cat(old_prob_target, dim=0)
        batch.old_prob_position = torch.cat(old_prob_position, dim=0)
        batch.old_prob_spell = torch.cat(old_prob_spell, dim=0)
        batch.returns = torch.tensor(
            batch.returns, dtype=torch.float, device=self.device
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
                act_logits, targets_logits, position_logits, spell_logits, value = self.ppo_net(**b.obs)
                act_logits = softmax(act_logits, mask_acts, self.device,add_little=True)
                dist_act = self.dist_fn(act_logits)
                prob_act = dist_act.log_prob(torch.tensor(b.act.act_id, device=self.device))
                targets_logits = softmax(targets_logits, mask_targets, self.device,add_little=True)
                dist_targets = self.dist_fn(targets_logits)
                prob_target = dist_targets.log_prob(torch.tensor(b.act.target_id, device=self.device))
                position_logits = softmax(position_logits, mask_position, self.device,add_little=True)
                dist_position = self.dist_fn(position_logits)
                prob_position = dist_position.log_prob(torch.tensor(b.act.position_id, device=self.device))
                spell_logits = softmax(spell_logits, mask_spell, self.device,add_little=True)
                dist_spell = self.dist_fn(spell_logits)
                prob_spell = dist_spell.log_prob(torch.tensor(b.act.spell_id, device=self.device))

                ratio = (prob_act + prob_target + prob_position + prob_spell - b.old_prob_act - b.old_prob_target - b.old_prob_position - b.old_prob_spell).exp().float()
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
                e_loss = (dist_act.entropy().mean() + dist_position.entropy().mean() + dist_targets.entropy().mean() + dist_spell.entropy().mean())/4
                ent_losses.append(e_loss.item())
                loss = clip_loss + self._w_vf * vf_loss - self._w_ent * e_loss
                losses.append(loss.item())
                self.optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ppo_net.parameters(),
                    self._max_grad_norm)
                self.optim.step()
        return {
            'loss': losses,
            'loss/clip': clip_losses,
            'loss/vf': vf_losses,
            'loss/ent': ent_losses,
        }
def collect_eps(agent,file,n_step = 200,n_episode = 1):
    buffer = ReplayBuffer(size=n_step,ignore_obs_next=True)
    battle = Battle(agent=agent)
    # battle.loadFile(f'ENV/debug{random.randint(1,2)}.json',shuffle_postion=True)
    battle.loadFile(file,shuffle_postion=True) #, shuffle_postion=True
    init_stack_position(battle)
    battle.checkNewRound()
    had_acted = False
    for ii in range(n_step):
        acting_stack = battle.curStack
        if acting_stack.by_AI:
            battle_action = acting_stack.active_stack()
            obs, acts, mask = None,None,None
        else:
            battle_action,obs,acts,mask = acting_stack.active_stack(ret_obs=True)
            had_acted = True
        battle.doAction(battle_action)
        battle.checkNewRound()
        #battle.curStack had updated
        done = battle.check_battle_end()
        reward = 0
        if done:
            reward = -1 if battle.by_AI[battle.get_winner()] else 1
            print("battle end")
            if acting_stack.by_AI:
                if had_acted:
                    buffer.rew[len(buffer) - 1] = reward
            else:
                buffer.add(obs=obs, act=acts, rew=reward, done=done, info=mask)
            break
        if not acting_stack.by_AI:
            buffer.add(obs=obs, act=acts, rew=reward, done=done,info=mask)
    if len(buffer) != 0:
        buffer.done[len(buffer) - 1] = True
    return buffer
def init_stack_position(battle):
    return
    mask = np.zeros([11,17])
    mask[:,0] = 1
    mask[:, 16] = 1
    me = battle.attacker_stacks[0]
    me.amountBase = random.randint(20,40)
    me.amount = me.amountBase
    for st in battle.stacks:
        pos = random.randint(1,11*17 - 1)
        while True:
            if mask[int(pos/17),pos%17]:
                pos = random.randint(1, 11 * 17 - 1)
            else:
                break
        st.x = pos%17
        st.y = int(pos/17)
        mask[st.y,st.x] = 1
def start_train():
    # 初始化 agent
    actor_critic = H3_net(dev)
    optim = torch.optim.Adam(actor_critic.parameters(), lr=1E-3)
    dist = torch.distributions.Categorical
    agent = H3_policy(actor_critic,optim,dist,device=dev)
    buffer = ReplayBuffer(2000,ignore_obs_next=True)
    file = 'env/debug1.json'
    while True:
        agent.eval()
        agent.in_train = False
        for _ in range(20):
            buffer_ep = collect_eps(agent,file)
            if len(buffer_ep):
                buffer.update(buffer_ep)
        batch_data, indice = buffer.sample(0)
        agent.train()
        agent.in_train = True
        batch_data = agent.process_gae(batch_data)
        loss = agent.learn(batch_data)
        print(loss)
        print(batch_data.done.astype(np.int))
        print(batch_data.rew.astype(np.int))
        print(batch_data.act.act_id.astype(np.int))
        agent.eval()
        agent.in_train = False
        cont = start_game(file,agent)
        if not cont:
            return
        buffer.reset()

def start_game(file,agent = None,by_AI = [0,1]):
    #初始化 agent
    if not agent:
        actor_critic = H3_net(dev)
        optim = torch.optim.Adam(actor_critic.parameters(), lr=1E-3)
        dist = torch.distributions.Categorical
        agent = H3_policy(actor_critic, optim, dist, device=dev)
    # 初始化游戏
    pygame.init()  # 初始化pygame
    pygame.display.set_caption('This is my first pyVCMI')  # 设置窗口标题
    # debug = True
    battle = Battle(debug=True,agent=agent,by_AI=by_AI)
    battle.loadFile(file,shuffle_postion=True)
    init_stack_position(battle)
    battle.checkNewRound()
    bi = H3_battleInterface.BattleInterface(battle)
    bi.next_act = battle.curStack.active_stack(print_act=True)
    act = bi.next_act
    # 事件循环(main loop)
    i = 0
    last = time.time()
    while bi.running:
        # if i % 1 == 0:
        if act:
            cost = time.time() - last
            print(f'cost time {cost}')
        act = bi.handleEvents()
        if act:
            i += 1
            last = time.time()
            if battle.check_battle_end():
                print("battle end")
                pygame.quit()
                return True
        bi.handleBattle(act,print_act = True)
        bi.renderFrame()
        # else:
        #     i += 1
        #     battle.doAction(bi.next_act)
        #     battle.checkNewRound()
        #     if battle.check_battle_end():
        #         print("battle end background")
        #         pygame.quit()
        #         return True
        #     bi.next_act = battle.curStack.active_stack()
    print("game end")
    pygame.quit()
    return False
def start_game_noGUI():
    #初始化 agent
    agent = H3_policy(3, 1024, device=dev)
    for ii in range(100):
        battle = Battle(agent=agent)
        battle.loadFile("ENV/selfplay.json")
        battle.checkNewRound()
        last = time.time()
        next_act = battle.curStack.active_stack()
        # 事件循环(main loop)
        i = 1
        while True:
            if i % 100 == 0:
                cost = time.time() - last
                print(f'cost time {cost}')
                last = time.time()
                break
            i += 1
            battle.doAction(next_act)
            battle.checkNewRound()
            if battle.check_battle_end():
                print("battle end")
                return
            next_act = battle.curStack.active_stack()

def main():
    start_train()
    # start_game(by_AI=[1, 1])
if __name__ == '__main__':
    main()