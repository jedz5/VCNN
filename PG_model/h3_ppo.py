
import sys
sys.path.extend(['/home/enigma/work/project/VCNN', 'D:/project/VCNN/'])
from PG_model.discrete_net import H3_net
import torch
from torch import nn
from typing import Tuple, Optional
from ENV.H3_battle import *
from VCbattle import BHex
from tianshou.policy import PGPolicy
from tianshou.data import Batch, ReplayBuffer
import scipy.signal
# import pdb

np.set_printoptions(precision=2,suppress=True,sign=' ',linewidth=400,formatter={'float': '{: 0.2f}'.format})
logger = get_logger()[1]
dev = 'cpu'
def softmax(logits, mask_orig,dev,add_little = False,in_train = True):
    mask = torch.tensor(mask_orig,dtype=torch.float,device=dev)
    if in_train:
        logits0 = (logits + 1E5) * mask
        # logits01 = (logits0.sub(logits0.min(dim=-1,keepdim=True)[0]))  # 保证>=0
        logits1 = logits0.sub(logits0.max(dim=-1,keepdim=True)[0]).exp()
        logits2 = logits1 * mask
        logits3 = logits2 / (torch.sum(logits2,dim=-1,keepdim=True) + 1E-9)
        if add_little:
            logits3 += 1E-9
    else:
        logits3 = (logits.sub(logits.min(dim=-1,keepdim=True)[0]) + 1)* mask
    return logits3
class H3_policy(PGPolicy):

    def __init__(self,
                 net: torch.nn.Module,
                 optim: torch.optim.Optimizer=None,
                 dist_fn: torch.distributions.Distribution=None,
                 device="cpu",
                 discount_factor: float = 0.95,
                 max_grad_norm: Optional[float] = 0.5,
                 eps_clip: float = .2,
                 vf_coef: float = .5,
                 ent_coef: float = .01,
                 action_range: Optional[Tuple[float, float]] = None,
                 gae_lambda: float = 1.,
                 dual_clip: float = None,
                 value_clip: bool = False,
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
        self._batch_size = 4000
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

    def process_gae(self, batch: Batch, single_batch=False,sil=False) -> Batch:
        # if self._rew_norm:
        #     mean, std = batch.rew.mean(), batch.rew.std()
        #     if std > self.__eps:
        #         batch.rew = (batch.rew - mean) / std
        # if self._lambda in [0, 1]:
        #     return self.compute_episodic_return(
        #         batch, None, gamma=self._gamma, gae_lambda=self._lambda)

        # return self.compute_episodic_return(
        #     batch, v_, gamma=self._gamma, gae_lambda=self._lambda)
        gamma = self._gamma
        gae_lambda = self._lambda

        if sil:
            v_ = []
            old_prob_act = []
            old_prob_position = []
            old_prob_target = []
            old_prob_spell = []
            with torch.no_grad():
                for b in batch.split(self._batch_size, shuffle=False):
                    act_logits, targets_logits_h,target_embs, position_logits_h, spell_logits, value = self.ppo_net(**b.obs)
                    v_.append(value.squeeze(-1))
                    act_index = torch.tensor(b.act.act_id, device=self.device, dtype=torch.long).unsqueeze(dim=-1)
                    targets_index = torch.tensor(b.act.target_id, device=self.device, dtype=torch.long).unsqueeze(dim=-1)

                    '''acts'''
                    act_logits = softmax(act_logits, b.info.mask_acts, self.device,add_little=True)
                    old_prob_act.append(act_logits.gather(dim=1,index=torch.tensor(b.act.act_id, device=self.device,dtype=torch.long).unsqueeze(-1)).squeeze(-1))
                    '''targets'''
                    targets_logits = self.ppo_net.get_target_loggits(act_index, targets_logits_h,single_batch=False)
                    targets_logits = softmax(targets_logits, b.info.mask_targets, self.device,add_little=True)
                    old_prob_target.append(targets_logits.gather(dim=1,index=torch.tensor(b.act.target_id, device=self.device,dtype=torch.long).unsqueeze(-1)).squeeze(-1))
                    '''position'''
                    position_logits = self.ppo_net.get_position_loggits(act_index, targets_index, target_embs,b.info.mask_targets,position_logits_h, single_batch=False)
                    position_logits = softmax(position_logits, b.info.mask_position, self.device,add_little=True)
                    old_prob_position.append(position_logits.gather(dim=1,index=torch.tensor(b.act.position_id, device=self.device,dtype=torch.long).unsqueeze(-1)).squeeze(-1))
                    '''spell'''
                    spell_logits = softmax(spell_logits, b.info.mask_spell, self.device,add_little=True)
                    old_prob_spell.append(spell_logits.gather(dim=1,index=torch.tensor(b.act.spell_id, device=self.device,dtype=torch.long).unsqueeze(-1)).squeeze(-1))

            batch.policy = Batch()
            batch.policy.logps = Batch()
            batch.policy.value = torch.cat(v_, dim=0).cpu().numpy()
            batch.policy.logps.act_logp = torch.cat(old_prob_act, dim=0).cpu().numpy()
            batch.policy.logps.target_logp = torch.cat(old_prob_target, dim=0).cpu().numpy()
            batch.policy.logps.position_logp = torch.cat(old_prob_position, dim=0).cpu().numpy()
            batch.policy.logps.spell_logp = torch.cat(old_prob_spell, dim=0).cpu().numpy()
            m = (1. - batch.done) * gamma
            rr = np.append(batch.rew, 0)
            vv = np.append(batch.policy.value, 0)
            deltas = rr[:-1] + m * vv[1:] - vv[:-1]
            mm = m * gae_lambda
            gae, ret = 0, 0
            adv = np.zeros_like(batch.rew,dtype=np.float32)
            returns = np.zeros_like(batch.rew,dtype=np.float32)
            for i in range(len(batch.rew) - 1, -1, -1):
                gae = deltas[i] + mm[i] * gae
                ret = batch.rew[i] + m[i] * ret
                adv[i] = gae
                returns[i] = ret
            batch.adv = adv
            batch.returns = returns
        else:
            if single_batch:
                rr = np.append(batch.rew, 0)
                vv = np.append(batch.policy.value, 0)
                delta = rr[:-1] + vv[1:] * gamma - vv[:-1]
                batch.adv = scipy.signal.lfilter([1], [1, -gamma * gae_lambda], delta[::-1], axis=0)[::-1]
                batch.returns = scipy.signal.lfilter([1], [1, -gamma], batch.rew[::-1], axis=0)[::-1]
            else:
                v_ = []
                with torch.no_grad():
                    for b in batch.split(self._batch_size, shuffle=False):
                        value = self.ppo_net(
                            **b.obs,critic_only=True)
                        v_.append(value.squeeze(-1))

                m = (1. - batch.done) * gamma
                rr = np.append(batch.rew, 0)
                vv = np.append(batch.policy.value, 0)
                deltas = rr[:-1] + m * vv[1:] - vv[:-1]
                mm = m * gae_lambda
                gae, ret = 0, 0
                adv = np.zeros_like(batch.rew, dtype=np.float32)
                returns = np.zeros_like(batch.rew, dtype=np.float32)
                for i in range(len(batch.rew) - 1, -1, -1):
                    gae = deltas[i] + mm[i] * gae
                    ret = batch.rew[i] + m[i] * ret
                    adv[i] = gae
                    returns[i] = ret
                batch.adv = adv
                batch.returns = returns


    #单次输入 obs->act,mask
    def forward(self, ind,attri_stack,planes_stack,plane_glb,
                env,can_shoot = False,print_act = False,action_known:tuple=None,
                **kwargs) -> Batch:
        act_logits, targets_logits_h,target_embs, position_logits_h, spell_logits, value = self.ppo_net(ind,attri_stack,planes_stack,plane_glb)
        '''ind,attri_stack,planes_stack,plane_glb shape like (batch,...) or (1,...) when inference'''
        mask_acts = env.legal_act(level=0)
        #act_id = -1
        act_logp = 1E-9
        mask_position = np.zeros(11 * 17, dtype=bool)
        position_id = -1
        position_logp = 1E-9
        mask_targets = np.zeros(14, dtype=bool)
        target_id = -1
        target_logp = 1E-9
        mask_spell = np.zeros(10, dtype=bool)
        spell_id = -1
        spell_logp = 1E-9
        value = value.item()
        if self.in_train:
            act_logits = softmax(act_logits, mask_acts, self.device,add_little=True)
            if print_act:
                logger.info(act_logits)
                logger.info(value)
            temp_p = self.dist_fn(act_logits)
            ''' act by human or handcrafted AI, just give me logp'''
            if action_known:
                act_id = action_known[0]
            else:
                act_id = temp_p.sample()[0].item()
            act_logp = act_logits[0,act_id].item()
            if act_id == action_type.move.value:
                mask_position = env.legal_act(level=1, act_id=act_id)
                '''target id is -1'''
                position_logits = self.ppo_net.get_position_loggits(act_id,target_id,target_embs,mask_targets,position_logits_h,single_batch=True)
                position_logits = softmax(position_logits, mask_position, self.device,add_little=True)
                temp_p = self.dist_fn(position_logits)
                if action_known:
                    position_id = action_known[1]
                else:
                    position_id = temp_p.sample()[0].item()
                position_logp = position_logits[0,position_id].item()
            elif act_id == action_type.attack.value:
                mask_targets = env.legal_act(level=1, act_id=act_id)
                targets_logits = self.ppo_net.get_target_loggits(act_id, targets_logits_h, single_batch=True)
                targets_logits = softmax(targets_logits, mask_targets, self.device,add_little=True)
                temp_p = self.dist_fn(targets_logits)
                if action_known:
                    target_id = action_known[2]
                else:
                    target_id = temp_p.sample()[0].item()
                target_logp = targets_logits[0,target_id].item()
                if not can_shoot:
                    mask_position = env.legal_act(level=2, act_id=act_id, target_id=target_id)
                    position_logits = self.ppo_net.get_position_loggits(act_id, target_id, target_embs, mask_targets,position_logits_h, single_batch=True)
                    position_logits = softmax(position_logits, mask_position, self.device,add_little=True)
                    temp_p = self.dist_fn(position_logits)
                    if action_known:
                        position_id = action_known[1]
                    else:
                        position_id = temp_p.sample()[0].item()
                    position_logp = position_logits[0,position_id].item()
        else:
            act_logits = softmax(act_logits, mask_acts, self.device,in_train=False)
            if print_act:
                logger.info(act_logits)
                logger.info(value)
            act_id = torch.argmax(act_logits,dim=-1)[0].item()
            if act_id == action_type.move.value:
                mask_position = env.legal_act(level=1, act_id=act_id)
                position_logits = self.ppo_net.get_position_loggits(act_id, target_id, target_embs, mask_targets,position_logits_h, single_batch=True)
                position_logits = softmax(position_logits, mask_position, self.device,in_train=False)
                position_id = torch.argmax(position_logits,dim=-1)[0].item()
            elif act_id == action_type.attack.value:
                mask_targets = env.legal_act(level=1, act_id=act_id)
                targets_logits = self.ppo_net.get_target_loggits(act_id, targets_logits_h, single_batch=True)
                targets_logits = softmax(targets_logits, mask_targets, self.device,in_train=False)
                target_id = torch.argmax(targets_logits, dim=-1)[0].item()
                if not can_shoot:
                    mask_position = env.legal_act(level=2, act_id=act_id, target_id=target_id)
                    position_logits = self.ppo_net.get_position_loggits(act_id, target_id, target_embs, mask_targets,position_logits_h, single_batch=True)
                    position_logits = softmax(position_logits, mask_position, self.device,in_train=False)
                    position_id = torch.argmax(position_logits, dim=-1)[0].item()
        return {'act_id': act_id, 'spell_id': spell_id, 'target_id': target_id, 'position_id': position_id,
                'mask_acts': mask_acts, 'mask_spell': mask_spell, 'mask_targets': mask_targets,'mask_position': mask_position,
                'act_logp':act_logp,'target_logp':target_logp,'position_logp':position_logp,'spell_logp':spell_logp,
                'value': value}

    #@profile
    #批量输入训练
    def learn(self, batch, batch_size=None, repeat=4, **kwargs):
        pcount = 0
        for _ in range(repeat):
            for b in batch.split(batch_size,shuffle=True):
                mask_acts = b.info.mask_acts
                mask_spell = b.info.mask_spell
                mask_targets = b.info.mask_targets
                mask_position = b.info.mask_position
                act_logits, targets_logits_h,target_embs, position_logits_h, spell_logits, value = self.ppo_net(**b.obs)
                act_index = torch.tensor(b.act.act_id, device=self.device, dtype=torch.long).unsqueeze(dim=-1)
                targets_index = torch.tensor(b.act.target_id, device=self.device, dtype=torch.long).unsqueeze(dim=-1)

                '''act gather softmax (batch,act_kinds) by (batch,1) -> (batch,1)'''
                act_logits = softmax(act_logits, mask_acts, self.device,add_little=True)
                prob_act = act_logits.gather(dim=1,index=act_index).squeeze(-1)
                dist_act = self.dist_fn(act_logits)
                # prob_act = dist_act.log_prob(torch.tensor(b.act.act_id, device=self.device)) * torch.tensor(mask_acts, device=self.device)

                '''target'''
                targets_logits = self.ppo_net.get_target_loggits(act_index, targets_logits_h, single_batch=False)
                targets_logits = softmax(targets_logits, mask_targets, self.device,add_little=True)
                prob_target = targets_logits.gather(dim=1,index=targets_index).squeeze(-1)
                dist_targets = self.dist_fn(targets_logits)
                # prob_target = dist_targets.log_prob(torch.tensor(b.act.target_id, device=self.device)) * torch.tensor(mask_targets, device=self.device)

                '''position'''
                position_logits = self.ppo_net.get_position_loggits(act_index, targets_index, target_embs, mask_targets,position_logits_h, single_batch=False)
                position_logits = softmax(position_logits, mask_position, self.device,add_little=True)
                prob_position = position_logits.gather(dim=1,index=torch.tensor(b.act.position_id, device=self.device,dtype=torch.long).unsqueeze(dim=-1)).squeeze(-1)
                dist_position = self.dist_fn(position_logits)
                # prob_position = dist_position.log_prob(torch.tensor(b.act.position_id, device=self.device)) * torch.tensor(mask_position, device=self.device)

                '''spell'''
                spell_logits = softmax(spell_logits, mask_spell, self.device,add_little=True)
                prob_spell = spell_logits.gather(dim=1,index=torch.tensor(b.act.spell_id, device=self.device,dtype=torch.long).unsqueeze(dim=-1)).squeeze(-1)
                dist_spell = self.dist_fn(spell_logits)
                # prob_spell = dist_spell.log_prob(torch.tensor(b.act.spell_id, device=self.device)) * torch.tensor(mask_spell, device=self.device)
                #TODO 如果old_prob太小？？？
                old_prob_act = torch.tensor(b.policy.logps.act_logp, device=self.device)
                old_prob_target = torch.tensor(b.policy.logps.target_logp, device=self.device)
                old_prob_position = torch.tensor(b.policy.logps.position_logp, device=self.device)
                old_prob_spell = torch.tensor(b.policy.logps.spell_logp, device=self.device)
                adv = torch.tensor(b.adv, device=self.device)
                returns = torch.tensor(b.returns, device=self.device)
                ratio = (prob_act / old_prob_act) * (prob_target / old_prob_target) * (prob_position / old_prob_position) * (prob_spell / old_prob_spell)

                surr1 = ratio * adv
                surr2 = ratio.clamp(
                    1. - self._eps_clip, 1. + self._eps_clip) * adv
                if self._dual_clip:
                    clip_loss = -torch.max(torch.min(surr1, surr2),
                                           self._dual_clip * adv).mean()
                else:
                    clip_loss = -torch.min(surr1, surr2).mean()
                # clip_losses.append(clip_loss.item())
                # if self._value_clip:
                #     v_clip = b.v + (value - b.v).clamp(
                #         -self._eps_clip, self._eps_clip)
                #     vf1 = (returns - value).pow(2)
                #     vf2 = (returns - v_clip).pow(2)
                #     vf_loss = .5 * torch.max(vf1, vf2).mean()
                # else:
                vf_loss = .5 * (returns - value.squeeze(-1)).pow(2).mean()
                # vf_losses.append(vf_loss.item())
                e_loss = dist_act.entropy().mean() + dist_position.entropy().mean() + dist_targets.entropy().mean() + dist_spell.entropy().mean()
                # ent_losses.append(e_loss.item())
                #TODO 9 reward 设计 32位 vs 64
                loss = clip_loss + self._w_vf * vf_loss - self._w_ent * e_loss
                if pcount < 5:
                    logger.info("clip_loss={:.4f} vf_loss={:.4f} e_loss={:.4f}".format(clip_loss,vf_loss,e_loss))
                pcount += 1
                # losses.append(loss.item())
                self.optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ppo_net.parameters(),
                    self._max_grad_norm)
                self.optim.step()
    def choose_action(self,in_battle:Battle,ret_obs = False,print_act=False,action_known:tuple=None):
        ind, attri_stack, planes_stack, plane_glb = in_battle.current_state_feature(curriculum=False)
        with torch.no_grad():
            result = self.forward(ind[None], attri_stack[None], planes_stack[None], plane_glb[None], in_battle,
                           can_shoot=in_battle.cur_stack.can_shoot(), print_act=print_act,action_known=action_known)
        act_id = result['act_id']
        position_id = result['position_id']
        target_id = result['target_id']
        spell_id = result['spell_id']
        if act_id == action_type.wait.value:
            next_act = BAction(action_type.wait)
        elif act_id == action_type.defend.value:
            next_act = BAction(action_type.defend)
        elif act_id == action_type.move.value:
            next_act = BAction(action_type.move,
                               dest=BHex(position_id % Battle.bFieldWidth, int(position_id / Battle.bFieldWidth)))
        elif act_id == action_type.attack.value:
            t = in_battle.stackQueue[target_id]
            next_act = BAction(action_type.attack,
                               dest=BHex(position_id % Battle.bFieldWidth, int(position_id / Battle.bFieldWidth)),
                               target=t)
        else:
            logger.error("not implemented action!!", True)
            sys.exit(-1)
        if not ret_obs:
            return next_act
        act_id = 0 if act_id < 0 else act_id
        position_id = 0 if position_id < 0 else position_id
        target_id = 0 if target_id < 0 else target_id
        spell_id = 0 if spell_id < 0 else spell_id
        obs = {'ind': ind, 'attri_stack': attri_stack, 'planes_stack': planes_stack, 'plane_glb': plane_glb}
        acts = {'act_id': act_id, 'position_id': position_id, 'target_id': target_id, 'spell_id': spell_id}
        mask = {'mask_acts': result['mask_acts'], 'mask_spell': result['mask_spell'],
                'mask_targets': result['mask_targets'], 'mask_position': result['mask_position']}
        logps = {'act_logp': result['act_logp'], 'position_logp': result['position_logp'],
                 'target_logp': result['target_logp'], 'spell_logp': result['spell_logp']}
        return next_act, obs, acts, mask, logps, result['value']
reward_def = 5
r_cum = 0
def record_sar(buffer,operator,battle,acting_stack,battle_action, obs, acts, mask, logps, value,done,killed_dealt, killed_get,td=True):
    global r_cum
    """compute reward"""
    reward = 0.
    tmp = 0.
    bl = len(buffer)
    if td:
        #TODO 是否需要判断win！！！！？？？
        if battle_action.type == action_type.attack:
            reward = killed_dealt * battle_action.target.ai_value / battle.ai_value[
                ~acting_stack.side] - killed_get * acting_stack.ai_value / battle.ai_value[acting_stack.side]
            if acting_stack.by_AI != operator:
                reward = -reward
        if done:
            if battle.by_AI[battle.get_winner()] == operator:  # winner is operator
                reward = 1. + reward
            else:
                reward = -1. + reward
        reward *= reward_def
        if done or acting_stack.by_AI == operator:
            tmp= r_cum
            r_cum = 0.
        else:
            r_cum += reward
        """buffer sar"""
        if acting_stack.by_AI == operator:
            if bl:
                buffer.rew[bl - 1] += tmp
            buffer.add(obs=obs, act=acts, rew=reward, done=done, info=mask, policy={"value": value, "logps": logps})
        elif done:
            if bl:
                buffer.rew[bl - 1] += reward
                buffer.done[bl - 1] = done
            else:
                logger.info("你的军队还没出手就被干掉了 (╯°Д°)╯︵┻━┻")
    else:
        if acting_stack.by_AI == operator:
            if done:
                if acting_stack.side == battle.get_winner():
                    buffer.add(obs=obs, act=acts, rew=reward_def, done=True, info=mask, policy={"value": value, "logps": logps})
                else:
                    buffer.add(obs=obs, act=acts, rew=-reward_def, done=True, info=mask,
                               policy={"value": value, "logps": logps})
            else:
                buffer.add(obs=obs, act=acts, rew=0, done=False, info=mask, policy={"value": value, "logps": logps})
        else:
            if done:
                if bl:
                    if acting_stack.side == battle.get_winner():
                        buffer.rew[bl - 1] = -reward_def
                        buffer.done[bl - 1] = True
                    else:
                        buffer.rew[bl - 1] = reward_def
                        buffer.done[bl - 1] = True
                else:
                    logger.info("你的军队还没出手就被干掉了 (╯°Д°)╯︵┻━┻")
global_buffer = ReplayBuffer(500,ignore_obs_next=True)
global_buffer_defender = ReplayBuffer(500,ignore_obs_next=True)
def collect_eps(agent,file=None,battle:Battle=None,n_step = 200,print_act = False,td=False):
    if not battle:
        battle = Battle(agent=agent)
        battle.load_battle(file)
    battle.checkNewRound()
    had_acted = False
    win = False
    acting_stack = battle.cur_stack
    if acting_stack.by_AI == 2:
        battle_action, obs, acts, mask, logps, value = acting_stack.active_stack(ret_obs=True, print_act=print_act)
    else:
        battle_action = acting_stack.active_stack()
        obs, acts, mask, logps, value = None, None, None, None, 0
    for ii in range(n_step):
        if acting_stack.by_AI == 2:
            if not had_acted:
                '''by now hand craft AI value is not needed'''
                # clear reward cumullated by hand craft AI
                # for st in battle.stacks:
                #     battle.ai_value[st.side] += st.ai_value * st.amount
                had_acted = True
        damage_dealt, damage_get, killed_dealt, killed_get = battle.doAction(battle_action)
        battle.checkNewRound()
        #battle.curStack had updated
        done = battle.check_battle_end()
        #buffer sar
        if had_acted:
            record_sar(global_buffer,2,battle,acting_stack,battle_action, obs, acts, mask, logps, value,done,killed_dealt, killed_get,td=td)
        #next act
        if done:
            win = (battle.by_AI[battle.get_winner()] == 2)
            break
        else:
            acting_stack = battle.cur_stack
            if acting_stack.by_AI == 2:
                print_act = print_act and ii < 5
                battle_action, obs, acts, mask, logps, value = acting_stack.active_stack(ret_obs=True,print_act=print_act)
            else:
                battle_action = acting_stack.active_stack()
                obs, acts, mask, logps, value = None, None, None, None, 0

    bat,indice = global_buffer.sample(0)
    global_buffer.reset()
    return win,bat
def to_action_tuple(action:BAction,battle:Battle):
    act_id = action.type.value
    dest_id = -1
    target_id = -1
    if action.type == action_type.move:
        dest_id = action.dest.y * Battle.bFieldWidth + action.dest.x
    if action.type == action_type.attack:
        target_id = 0
        find = False
        for st in battle.stackQueue:
            if st == action.target:
                find = True
                break
            else:
                target_id += 1
        assert find,f"where is your target{target}??"
        if not battle.cur_stack.can_shoot():
            dest_id = action.dest.y * Battle.bFieldWidth + action.dest.x
    return (act_id,dest_id,target_id)
def collect_eps_both_sides(agent,file,n_step = 200,print_act = False):
    battle = Battle(agent=agent)
    battle.load_battle(file)
    battle.checkNewRound()
    win = -1
    for ii in range(n_step):
        acting_stack = battle.cur_stack
        if acting_stack.by_AI == 2:
            battle_action, obs, acts, mask, logps, value = acting_stack.active_stack(ret_obs=True, print_act=print_act)
        else:
            orig_battle_action = acting_stack.active_stack()
            action_tuple = to_action_tuple(orig_battle_action,battle)
            '''just give me logp '''
            acting_stack.by_AI = 2
            battle_action, obs, acts, mask, logps, value = acting_stack.active_stack(ret_obs=True, print_act=print_act,
                                                                                     action_known=action_tuple)
            acting_stack.by_AI = 1
            '''gotcha'''
        battle.doAction(battle_action)
        battle.checkNewRound()
        #battle.curStack had updated
        done = battle.check_battle_end()
        #buffer sar
            # record_sar(global_buffer,2,battle,acting_stack,battle_action, obs, acts, mask, logps, value,done,killed_dealt, killed_get,td=False)
        if acting_stack.side == 0:
            global_buffer.add(obs=obs, act=acts, rew=int(done), done=int(done), info=mask, policy={"value": value, "logps": logps})
        else:
            global_buffer_defender.add(obs=obs, act=acts, rew=int(done), done=int(done), info=mask, policy={"value": value, "logps": logps})
        #next act
        if done:
            bla = len(global_buffer)
            bld = len(global_buffer_defender)
            win = battle.get_winner()
            if win == 0:
                if bla > 0:
                    global_buffer.rew[bla - 1] = reward_def
                    global_buffer.done[bla - 1] = 1
                else:
                    logger.info("玩家0还没出手就赢了~")
                if bld > 0:
                    global_buffer_defender.rew[bld - 1] = -reward_def
                    global_buffer_defender.done[bld - 1] = 1
                else:
                    logger.info("玩家1还没出手就输了~")
            else:
                if bla > 0:
                    global_buffer.rew[bla - 1] = -reward_def
                    global_buffer.done[bla - 1] = 1
                else:
                    logger.info("玩家0还没出手就输了~")
                if bld > 0:
                    global_buffer_defender.rew[bld - 1] = reward_def
                    global_buffer_defender.done[bld - 1] = 1
                else:
                    logger.info("玩家1还没出手就赢了~")
            break

    bat0,indice = global_buffer.sample(0)
    bat1,indice = global_buffer_defender.sample(0)
    global_buffer.reset()
    global_buffer_defender.reset()
    return win,bat0,bat1


def hook_me(grad):
    print(grad.abs().sum(dim=-1))
def init_stack_position(battle):
    mask = np.zeros([11,17])
    mask[:,0] = 1
    mask[:, 16] = 1
    for st in battle.stacks:
        # base1 = random.random() * 2 + 0.1
        # st.amount_base = int(st.amount_base * base1)
        # st.amount = st.amount_base
        pos = random.randint(1,11*17 - 1)
        while True:
            if mask[int(pos/17),pos%17]:
                pos = random.randint(1, 11 * 17 - 1)
            else:
                break
        st.x = pos%17
        st.y = int(pos/17)
        mask[st.y,st.x] = 1

def to_dev(agent,dev):
    agent.to(dev)
    agent.device = dev
    agent.ppo_net.device = dev
    agent.ppo_net.inpipe.device = dev
def start_game(file,battle_int=None,agent = None,by_AI = [2,1]):
    #初始化 agent
    if not agent:
        actor_critic = H3_net(dev)
        optim = None #torch.optim.Adam(actor_critic.parameters(), lr=1E-3)
        dist = torch.distributions.Categorical
        agent = H3_policy(actor_critic, optim, dist, device=dev)
        agent.in_train = True
    # debug = True
    battle = Battle(agent=agent,by_AI=by_AI)
    battle.load_battle(file,shuffle_postion=False)
    # if random.randint(0, 1):
    #     init_stack_position(battle)
    # init_stack_position(battle)
    battle.checkNewRound()
    if not battle_int:
        from ENV import H3_battleInterface
        battle_int = H3_battleInterface.BattleInterface(battle)
    else:
        battle_int.init_battle(battle)
    battle_int.next_act = battle.cur_stack.active_stack(print_act=True)
    i = 0
    while battle_int.running:
        act = battle_int.handleEvents()
        if act:
            i += 1
            if battle.check_battle_end():
                print("battle end")
                battle_int.running = False
                # pygame.quit()
                return 1 if battle.by_AI[battle.get_winner()] == 2 else 0
        battle_int.handleBattle(act,print_act = True)
        battle_int.renderFrame()
    print("game end")
    # pygame.quit()
    return 0
def test_game_noGUI(file,agent = None,by_AI = [2,1]):
    #初始化 agent
    test_win = 0
    iter_N = 3
    for ii in range(iter_N):
        battle = Battle(agent=agent)
        battle.load_battle(file)
        battle.checkNewRound()
        next_act = battle.cur_stack.active_stack()
        # 事件循环(main loop)
        while True:
            battle.doAction(next_act)
            battle.checkNewRound()
            if battle.check_battle_end():
                winner = battle.get_winner()
                logger.debug(f"battle end, winner is {winner}")
                if battle.by_AI[winner] == 2:
                    test_win += 1
                break
            next_act = battle.cur_stack.active_stack()
    return test_win/iter_N
def get_act_info(battle,act):
    target_id = -1
    position_id = -1
    act_id = act.type.value
    mask_acts, mask_spell, mask_targets, mask_position = battle.get_act_masks(act)
    ind, attri_stack, planes_stack, plane_glb = battle.current_state_feature(curriculum=False)
    if act.type == action_type.attack:
        if not battle.cur_stack.can_shoot():
            position_id = act.dest.to_position_id()
        else:
            position_id = 0
        tgt_stacks = battle.attacker_stacks if battle.cur_stack.side else battle.defender_stacks
        for i, st in enumerate(tgt_stacks):
            if st.is_alive() and st == act.target:
                target_id = i
        if target_id < 0:
            logger.error("no target found")
            sys.exit(-1)
    if act.type == action_type.move:
        target_id = 0
        position_id = act.dest.to_position_id()
    if act.type == action_type.wait:
        target_id = 0
        position_id = 0
    if act.type == action_type.defend:
        target_id = 0
        position_id = 0
    obs = {'ind': ind, 'attri_stack': attri_stack, 'planes_stack': planes_stack, 'plane_glb': plane_glb}
    acts = {'act_id': act_id, 'position_id': position_id, 'target_id': target_id, 'spell_id': 0}
    mask = {'mask_acts': mask_acts, 'mask_spell': mask_spell,
            'mask_targets': mask_targets, 'mask_position': mask_position}
    return obs,acts,mask
def start_game_record(battle:Battle=None):
    from tianshou.data import ReplayBuffer
    from ENV import H3_battleInterface
    import pygame
    buffer = ReplayBuffer(500,ignore_obs_next=True)
    # 初始化游戏
    pygame.init()  # 初始化pygame
    pygame.display.set_caption('This is my first pyVCMI')  # 设置窗口标题
    if not battle:
        battle = Battle(by_AI = [0,1])
        battle.load_battle("ENV/battles/1.json", shuffle_postion=False,load_ai_side=False)
    battle.checkNewRound()
    bi = H3_battleInterface.BattleInterface(battle)
    logps, value = None,0
    had_acted = False
    acting_stack = battle.cur_stack
    bi.next_act = acting_stack.active_stack()
    # 事件循环(main loop)
    while bi.running:
        do_act = bi.handleEvents()
        if do_act:
            if acting_stack.by_AI == 0:
                if not had_acted:
                    for st in battle.stacks:
                        battle.ai_value[st.side] += st.ai_value * st.amount
                    had_acted = True
                obs, acts, mask = get_act_info(battle, bi.next_act)
            else:
                obs, acts, mask = None, None, None
            damage_dealt, damage_get, killed_dealt, killed_get = battle.doAction(bi.next_act)
            done = battle.check_battle_end()
            if had_acted:
                record_sar(buffer, 0, battle, acting_stack, bi.next_act, obs, acts, mask, logps, value, done,
                          killed_dealt,
                          killed_get,td=False)
            battle.checkNewRound()
            if done:
                logger.debug("battle end~")
                bi.running = False
                # pygame.quit()
                # break
            else:
                acting_stack = battle.cur_stack
                bi.next_act = acting_stack.active_stack()
        bi.renderFrame()
    data, indice = buffer.sample(0)
    return data
    # np.save("d:/xxx.npy", [data.obs, data.act, data.rew,data.done,data.info])
def dump_episo(ep,dir):
    files = []
    for f in os.listdir(dir):
        tmp_f = os.path.join(dir,f)
        if os.path.isdir(tmp_f):
            continue
        else:
            files.append(int(os.path.splitext(f)[0]))
    if len(files):
        nmax = max(files) + 1
    else:
        nmax = 0
    dump_in = os.path.join(dir, f'{nmax}.npy')
    np.save(dump_in,ep)
    print(f"episode dumped in {dump_in}")
def load_episo(dir):
    if len(os.listdir(dir)) == 0:
        return
    obss,obs_nexts,acts,masks,rews,dones =[],[],[],[],[],[]
    for f in os.listdir(dir):
        tmp_f = os.path.join(dir,f)
        if os.path.isdir(tmp_f):
            continue
        else:
            obs,obs_next, act, rew, done, mask = np.load(tmp_f, allow_pickle=True)
            obs = Batch.stack(obs)
            obs_next = Batch.stack(obs_next)
            act = Batch.stack(act)
            mask = Batch.stack(mask)
            obss.append(obs)
            obs_nexts.append(obs_next)
            acts.append(act)
            masks.append(mask)
            rews.append(rew)
            dones.append(done)
    obs2 = Batch.cat(obss)
    obs_next2 = Batch.cat(obs_nexts)
    act2 = Batch.cat(acts)
    mask2 = Batch.cat(masks)
    rew2 = np.concatenate(rews)
    done2 = np.concatenate(dones)
    expert = Batch(obs=obs2,obs_next = obs_next2,act=act2, rew=rew2, done=done2,info=mask2,policy = Batch())
    return expert
def start_test():
    import pygame
    pygame.init()  # 初始化pygame
    pygame.display.set_caption('This is my first pyVCMI')  # 设置窗口标题

    actor_critic = H3_net(dev)
    optim = None #torch.optim.Adam(actor_critic.parameters(), lr=0.005)
    dist = torch.distributions.Categorical
    agent = H3_policy(actor_critic, optim, dist, device=dev)
    agent.load_state_dict(torch.load("model_param.pkl"))
    agent.eval()
    agent.in_train = False
    start_game("ENV/battles/6.json", by_AI=[2, 1],agent=agent)
#@profile
def start_train():
    lrate = 0.0005
    sample_num = 100
    # 初始化 agent
    actor_critic = H3_net(dev)
    optim = torch.optim.Adam(actor_critic.parameters(), lr=lrate)
    # actor_critic.act_ids.weight.register_hook(hook_me)
    dist = torch.distributions.Categorical
    agent = H3_policy(actor_critic,optim,dist,device=dev,gae_lambda=0.95)
    # agent.load_state_dict(torch.load("model_param.pkl"))
    count = 0
    five_done = 5
    ok = five_done
    max_sar = [None] * 4 #type:List[Batch]
    expert = load_episo("ENV/episode")
    if expert:
        max_sar[0] = Batch(expert)
        cumulate_reward(expert)
    file_list = ['ENV/battles/0.json','ENV/battles/1.json','ENV/battles/2.json','ENV/battles/3.json']

    file_list_cache = []
    '''cache json'''
    for file in file_list:
        arena = Battle()
        arena.load_battle(file,load_ai_side=False,format_postion=True)
        arena.checkNewRound()
        start = arena.current_state_feature(curriculum=True)
        file_list_cache.append((start,arena.round))
    cache_idx = range(len(file_list))

    def choose_start(idx):
        if not max_sar[idx] or np.random.binomial(1,0.2):
            return -1,file_list_cache[idx] #RL
        else:
            obs_idx = random.choice(range(len(max_sar[idx].obs)))  # SIL
            return obs_idx,(max_sar[idx].obs[obs_idx].attri_stack,0) #FIXME round = 0
    def update_max(idx,obs_idx,data:Batch):
        if (not max_sar[idx]):
            max_sar[idx] = Batch(data)
        elif (sum(max_sar[idx].rew[obs_idx:]) < sum(data.rew)):
            max_sar[idx] = Batch.cat([max_sar[idx][:obs_idx],Batch(data)])
    max_win_count = len(file_list)
    while True:
        agent.eval()
        agent.in_train = True
        bats = []
        for exp in max_sar:
            if exp:
                exp_copy = Batch(exp)
                agent.process_gae(exp_copy, single_batch=False, sil=True)
                bats.append(exp_copy)
        for ii in range(sample_num):
            file_idx = random.choice(cache_idx)
            print_act = False
            # if ii < 3 :
            #     print_act = True
            #     logger.info(f"------------------------{fn}.json")
            '''single side '''
            # win,batch_data = collect_eps(agent,file,print_act=print_act)
            # agent.process_gae(batch_data)
            # bats.append(batch_data)
            '''both sides'''
            # win, batch0,batch1 = collect_eps_both_sides(agent, file_list_cache[file_idx], print_act=print_act)
            # agent.process_gae(batch0)
            # bats.append(batch0)
            # agent.process_gae(batch1)
            # bats.append(batch1)
            # if win == 0 and ii < 50:
            #     logger.info(f"win {file_list[file_idx]}")
            '''single side Wheel fight'''
            arena = Battle(by_AI=[2, 1],agent=agent)
            obs_idx,obs = choose_start(file_idx)
            arena.load_battle(obs, load_ai_side=False, format_postion=True)
            wins = []
            for r in range(10):
                win,batch_data = collect_eps(agent,battle=arena,print_act=print_act)
                if win:
                    wins.append(batch_data)
                    arena.reset()
                    arena.split_army()
                else:
                    bats.append(batch_data)
                    break
            if len(wins):
                wins_batch = Batch.cat(wins)
                update_max(file_idx,obs_idx,wins_batch)
                cumulate_reward(wins_batch)
                bats.append(wins_batch)


        batch_data = Batch.cat(bats)
        logger.info(len(batch_data.rew))
        agent.train()
        agent.in_train = True
        agent.process_gae(batch_data,single_batch=False)
        # v_ = []
        # with torch.no_grad():
        #     v_ = agent.ppo_net(**batch_data.obs, critic_only=True)
        # print(batch_data.returns)
        # v_ = v_.squeeze().detach().numpy()
        # print("predict v")
        # print(v_)
        # print("returns - v")
        # print(batch_data.returns - v_)

        logger.info(batch_data.done.astype(np.float)[:35] * 1.11)
        logger.info("act_logp")
        # idx = np.random.choice(range(len(batch_data.rew)),size=20)
        logger.info(batch_data.policy.logps.act_logp[:35])
        logger.info(batch_data.act.act_id.astype(np.float)[:35] )
        logger.info("position_logp")
        logger.info(batch_data.policy.logps.position_logp[:35])
        logger.info(batch_data.act.position_id.astype(np.float)[:35] //17)
        logger.info(batch_data.act.position_id.astype(np.float)[:35] % 17 - 9)
        logger.info("target")
        logger.info(batch_data.act.target_id.astype(np.float)[:35])
        logger.info("adv")
        logger.info(batch_data.adv[:35])
        logger.info(batch_data.policy.value[:35])
        if Linux:
            to_dev(agent, "cuda")
        loss = agent.learn(batch_data,batch_size=2000)
        if Linux:
            to_dev(agent, "cpu")
        agent.eval()
        agent.in_train = False

        '''test by win rate'''
        # win_rate = 0
        # for file_idx in cache_idx:
        #     ct = test_game_noGUI(file_list_cache[file_idx], agent=agent)
        #     logger.info(f"test-{count}-{file_list[file_idx]} win rate = {ct}")
        #     win_rate += ct
        # win_rate /= len(file_list)
        # logger.info(f"win rate at all = {win_rate}")
        # if win_rate > 0.9:
        #     ok -= 1
        #     if ok == 0:
        #         torch.save(agent.state_dict(),"model_param.pkl")
        #         logger.info("model saved")
        #         sys.exit(1)
        # else:
        #     ok = five_done
        # if count == 50000:
        #     sys.exit(-1)
        '''test how many times agent can win'''
        win_count = []
        for file_idx in cache_idx:
            arena = Battle(by_AI=[2, 1], agent=agent)
            arena.load_battle(file_list_cache[file_idx], load_ai_side=False, format_postion=True)
            ct = 0
            for r in range(10):
                arena.split_army()
                win, batch_data = collect_eps(agent, battle=arena, print_act=print_act)
                if win:
                    ct += 1
                    arena.reset()
                else:
                    win_count.append(ct)
                    logger.info(f"test-{count}-{file_list[file_idx]} win count = {ct}")
                    break
        if sum(win_count) > max_win_count:
            logger.info("model saved")
            torch.save(agent.state_dict(), "model_param.pkl")
            max_win_count = sum(win_count)
        count += 1
        logger.info(f'count={count}')

def cumulate_reward(batch:Batch):
    a = batch.rew
    s = a.sum()
    lk = np.array(range(s - reward_def, -1, -reward_def))
    a[batch.done > 0.5] += lk
M=0
if __name__ == '__main__':
    # start_game_record()
    start_train()
    # start_test()

    # arena = Battle(by_AI=[0, 1])
    # arena.load_battle("ENV/battles/0.json", load_ai_side=False, format_postion=True)
    # arena.split_army()
    # arena.checkNewRound()
    # data1 = start_game_record(battle=arena)
    # #
    # arena.reset()
    # arena.split_army()
    # arena.checkNewRound()
    # data2 = start_game_record(battle=arena)
    # data = Batch.cat([data1,data2])
    # dump_episo([data.obs, data.obs_next, data.act, data.rew, data.done, data.info], "ENV/episode")
    # data = load_episo("ENV/episode")
    # cumulate_reward(data)