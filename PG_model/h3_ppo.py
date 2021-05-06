
import sys
sys.path.extend(['/home/enigma/work/project/VCNN', 'D:/project/VCNN/'])
from PG_model.discrete_net import H3_net
import torch
from torch import nn
from typing import Tuple, Optional
from ENV.H3_battle import *
from PG_model.discrete_net import *
from VCbattle import BHex
from tianshou.policy import PGPolicy
from tianshou.data import Batch, ReplayBuffer
import scipy.signal
import torch.nn.functional as F
# import pdb

np.set_printoptions(precision=1,suppress=True,sign=' ',linewidth=400,formatter={'float': '{: 0.1f}'.format})
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
class H3AgentQ(nn.Module):

    def __init__(self,
                 optim: torch.optim.Optimizer=None,
                 device="cpu",
                 discount_factor: float = 0.95,
                 threshold = 0.3,
                 eps_clip: float = .2,
                 vf_coef: float = .5,
                 p_explore = 0.3,
                 **kwargs) -> None:
        super().__init__()
        self._eps_clip = eps_clip
        self._w_vf = vf_coef
        self.threshold = threshold
        self.discount_factor = discount_factor
        self.net = H3_Q_net()
        self.target_net = self.net.clone()
        self.optim = optim
        self._batch_size = 512
        self.p_explore = p_explore
        self.device = device
    def f_act(self,act_q,act_imt,mask_acts):
        act_imt = act_imt.softmax(dim=-1)  # shape(1,5)
        imt = act_imt / act_imt.max(1, keepdim=True)[0] > self.threshold  # shape(1,5)
        imt = torch.logical_and(imt,
                                torch.tensor(mask_acts, dtype=bool).unsqueeze(0)).float()  # shape(1,5) && shape(1,5)
        # Use large negative number to mask actions from argmax
        act_id = int((imt * act_q + (1. - imt) * -1e8).argmax(1))
        return act_id
    def f_target(self,act_id, targets_logits_h,mask_targets, single_batch=True):
        targets_q, targets_imt = self.net.get_target_q(act_id, targets_logits_h, single_batch=single_batch)
        targets_imt = targets_imt.softmax(dim=-1)  # shape(1,5)
        imt = targets_imt / targets_imt.max(1, keepdim=True)[0] > self.threshold  # shape(1,5)
        imt = torch.logical_and(imt,torch.tensor(mask_targets, dtype=bool).unsqueeze(0)).float()  # shape(1,5) && shape(1,5)
        # Use large negative number to mask actions from argmax
        target_id = int((imt * targets_q + (1. - imt) * -1e8).argmax(1))
        return target_id
    def f_position(self,act_id, target_id, target_embs, mask_targets,position_logits_h,mask_position, single_batch=True):
        '''target id is -1 mask_targets = all 0'''
        position_q, position_imt = self.net.get_position_q(act_id, target_id, target_embs, mask_targets,
                                                                 position_logits_h, single_batch=single_batch)
        position_imt = position_imt.softmax(dim=-1)  # shape(1,5)
        imt = position_imt / position_imt.max(1, keepdim=True)[0] > self.threshold  # shape(1,5)
        imt = torch.logical_and(imt, torch.tensor(mask_position, dtype=bool).unsqueeze(0)).float()  # shape(1,5) && shape(1,5)
        position_id = int((imt * position_q + (1. - imt) * -1e8).argmax(1))
        return position_id
    #单次输入 obs->act,mask
    def forward(self, ind,attri_stack,planes_stack,plane_glb,env,can_shoot = False,print_act = False) -> Batch:
        act_q, act_imt, targets_logits_h,target_embs, position_logits_h, spell_logits = self.net(ind,attri_stack,planes_stack,plane_glb)
        '''ind,attri_stack,planes_stack,plane_glb shape like (batch,...) or (1,...) when inference'''

        if self.in_train:
            mask_acts = env.legal_act(level=0)
            if np.random.binomial(1,self.p_explore):
                act_id = np.random.choice(list(range(len(mask_acts))), p=mask_acts / mask_acts.sum())
            else:
                act_id = self.f_act(act_q,act_imt,mask_acts)
            if act_id == action_type.move.value:
                '''no target'''
                mask_targets = np.zeros(14, dtype=bool)
                target_id = -1
                '''position only'''
                mask_position = env.legal_act(level=1, act_id=act_id)
                if np.random.binomial(1, self.p_explore):
                    position_id = np.random.choice(list(range(len(mask_position))), p=mask_position / mask_position.sum())
                else:
                    position_id = self.f_position(act_id, target_id, target_embs, mask_targets,position_logits_h,mask_position, single_batch=True)
            elif act_id == action_type.attack.value:
                mask_targets = env.legal_act(level=1, act_id=act_id)
                if np.random.binomial(1, self.p_explore):
                    target_id = np.random.choice(list(range(len(mask_targets))), p=mask_targets / mask_targets.sum())
                else:
                    target_id = self.f_target(act_id, targets_logits_h, mask_targets, single_batch=True)
                if can_shoot:
                    '''no position'''
                    mask_position = np.zeros(11 * 17, dtype=bool)
                    position_id  = -1
                else:
                    mask_position = env.legal_act(level=2, act_id=act_id, target_id=target_id)
                    if np.random.binomial(1, self.p_explore):
                        position_id = np.random.choice(list(range(len(mask_position))),p=mask_position / mask_position.sum())
                    else:
                        position_id = self.f_position(act_id, target_id, target_embs, mask_targets, position_logits_h,mask_position, single_batch=True)
        else:
            mask_acts = env.legal_act(level=0)
            act_id = self.f_act(act_q, act_imt, mask_acts)
            if act_id == action_type.move.value:
                '''no target'''
                mask_targets = np.zeros(14, dtype=bool)
                target_id = -1
                '''position only'''
                mask_position = env.legal_act(level=1, act_id=act_id)
                position_id = self.f_position(act_id, target_id, target_embs, mask_targets, position_logits_h,mask_position, single_batch=True)
            elif act_id == action_type.attack.value:
                mask_targets = env.legal_act(level=1, act_id=act_id)
                target_id = self.f_target(act_id, targets_logits_h, mask_targets, single_batch=True)
                if can_shoot:
                    '''no position'''
                    mask_position = np.zeros(11 * 17, dtype=bool)
                    position_id = -1
                else:
                    mask_position = env.legal_act(level=2, act_id=act_id, target_id=target_id)
                    position_id = self.f_position(act_id, target_id, target_embs, mask_targets, position_logits_h,mask_position, single_batch=True)
        return {'act_id': act_id, 'spell_id': -1, 'target_id': target_id, 'position_id': position_id,
                'mask_acts': mask_acts, 'mask_targets': mask_targets,'mask_position': mask_position}

    #@profile
    #批量输入训练
    def learn(self, batch, batch_size=None, repeat=4, **kwargs):
        pcount = 0
        for _ in range(repeat):
            for b in batch.split(batch_size,shuffle=True):
                mask_acts = b.info.mask_acts
                mask_targets = b.info.mask_targets
                mask_position = b.info.mask_position
                act_index = torch.tensor(b.act.act_id, device=self.device, dtype=torch.long).unsqueeze(dim=-1)
                targets_index = torch.tensor(b.act.target_id, device=self.device, dtype=torch.long).unsqueeze(dim=-1)

                current_act_q, current_act_imt, targets_logits_h, target_embs, position_logits_h, spell_logits = self.net(**b.obs)
                # target_act_id = self.f_act(current_act_q, current_act_imt,mask_acts)
                current_targets_q, current_targets_imt = self.net.get_target_q(act_index, targets_logits_h, single_batch=False)
                target_targets_id = self.f_target(act_index,)
                target_targets_q, target_targets_imt = self.target_net.get_target_q(act_index, targets_logits_h, single_batch=False)
                current_position_q, current_position_imt = self.net.get_position_q(act_index, targets_index, target_embs, mask_targets,position_logits_h, single_batch=False)
                target_position_q, target_position_imt = self.target_net.get_position_q(act_index, targets_index, target_embs, mask_targets,position_logits_h, single_batch=False)
                # with torch.no_grad():
                #     target_act_q_next, target_act_imt_next, targets_logits_h_next, target_embs_next, position_logits_h_next, spell_logits_next = self.net(**b.obs_next)
                #
                #
                #
                # #TODO 9 reward 设计 32位 vs 64
                # loss = clip_loss + self._w_vf * vf_loss - self._w_ent * e_loss
                # if pcount < 5:
                #     logger.info("clip_loss={:.4f} vf_loss={:.4f} e_loss={:.4f}".format(clip_loss,vf_loss,e_loss))
                # pcount += 1
                # # losses.append(loss.item())
                # self.optim.zero_grad()
                # loss.backward()
                # nn.utils.clip_grad_norm_(self.net.parameters(),
                #     self._max_grad_norm)
                # self.optim.step()
    def choose_action(self,in_battle:Battle,ret_obs = False,print_act=False,action_known:tuple=None):
        ind, attri_stack, planes_stack, plane_glb = in_battle.current_state_feature(curriculum=False)
        with torch.no_grad():
            result = self.forward(ind[None], attri_stack[None], planes_stack[None], plane_glb[None], in_battle,
                           can_shoot=in_battle.cur_stack.can_shoot(), print_act=print_act)
        act_id = result['act_id']
        position_id = result['position_id']
        target_id = result['target_id']
        spell_id = result['spell_id']
        next_act = BAction.idx_to_action(act_id,position_id,target_id,spell_id,in_battle)
        if not ret_obs:
            return next_act
        act_id = 0 if act_id < 0 else act_id
        position_id = 0 if position_id < 0 else position_id
        target_id = 0 if target_id < 0 else target_id
        spell_id = 0 if spell_id < 0 else spell_id
        obs = {'ind': ind, 'attri_stack': attri_stack, 'planes_stack': planes_stack, 'plane_glb': plane_glb}
        acts = {'act_id': act_id, 'position_id': position_id, 'target_id': target_id, 'spell_id': spell_id}
        mask = {'mask_acts': result['mask_acts'], 'mask_targets': result['mask_targets'], 'mask_position': result['mask_position']}
        return next_act,{"act":acts,"obs":obs,"mask":mask}
class H3ReplayManager:
    def __init__(self,file_list,agent,p_start_from_scratch = 0.2,use_expert_data=False, format_postion=False):
        fl = len(file_list)
        self.win_rate = [0] * fl
        self.start_point = [0] * fl
        self.max_sar = [None] * fl #type:List[Batch]
        self.agent = agent
        self.use_expert_data = use_expert_data
        if use_expert_data:
            self.init_expert_data()
        self.p_start_from_scratch = p_start_from_scratch
    def init_expert_data(self):
        expert = load_episo("ENV/episode")
        if expert:
            self.start_point[0] = len(expert) - 1
            self.max_sar[0] = Batch(expert,copy=True)
            H3SampleCollector.cumulate_reward(expert.rew)
            self.agent.process_gae(expert, single_batch=False, sil=True)
            '''fill in policy'''
            self.max_sar[0].policy = expert.policy
    def choose_start(self,idx,from_start=False):
        if not self.max_sar[idx] or from_start or np.random.binomial(1,self.p_start_from_scratch):
            return -1,self.file_list_cache[idx],None #RL
        else:
            obs_idx = random.choice(range(len(self.max_sar[idx].obs)))  # SIL
            return obs_idx,(self.max_sar[idx].obs[obs_idx].attri_stack,0),[copy.copy(st) for st in self.defender_states[idx]] #FIXME round = 0
    def update_max(self,idx,obs_idx,data:ReplayBuffer,data_rew,indice):
        if obs_idx < 0:
            if (not self.max_sar[idx] and data_rew > 0):
                self.max_sar[idx] = data[indice]
            elif self.max_sar[idx] and (sum(self.max_sar[idx].rew) < data_rew):
                self.max_sar[idx] = data[indice]
        elif (sum(self.max_sar[idx].rew[obs_idx:]) < data_rew):
            self.max_sar[idx] = Batch.cat([self.max_sar[idx][:obs_idx], data[indice]])
            max_data = self.max_sar[idx]
            dump_episo([max_data.obs, max_data.act, max_data.rew, max_data.done, max_data.info], "ENV/max_sars",f'max_data_{idx}.npy')
            logger.info(f"states dumped in max_data_{idx}.npy")
    def get_sars(self):
        bats = []
        for exp in self.max_sar:
            if exp:
                exp_copy = Batch(exp,copy=True)
                H3SampleCollector.cumulate_reward(exp_copy.rew)
                self.agent.process_gae(exp_copy, single_batch=False, sil=True)
                bats.append(exp_copy)
                logger.info(f"exp.rew={sum(exp.rew)}")
        return bats
class H3Agent(PGPolicy):

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
        self.net = net
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
                    act_logits, targets_logits_h,target_embs, position_logits_h, spell_logits, value = self.net(**b.obs)
                    v_.append(value.squeeze(-1))
                    act_index = torch.tensor(b.act.act_id, device=self.device, dtype=torch.long).unsqueeze(dim=-1)
                    targets_index = torch.tensor(b.act.target_id, device=self.device, dtype=torch.long).unsqueeze(dim=-1)

                    '''acts'''
                    act_logits = softmax(act_logits, b.info.mask_acts, self.device,add_little=True)
                    old_prob_act.append(act_logits.gather(dim=1,index=torch.tensor(b.act.act_id, device=self.device,dtype=torch.long).unsqueeze(-1)).squeeze(-1))
                    '''targets'''
                    targets_logits = self.net.get_target_loggits(act_index, targets_logits_h,single_batch=False)
                    targets_logits = softmax(targets_logits, b.info.mask_targets, self.device,add_little=True)
                    old_prob_target.append(targets_logits.gather(dim=1,index=torch.tensor(b.act.target_id, device=self.device,dtype=torch.long).unsqueeze(-1)).squeeze(-1))
                    '''position'''
                    position_logits = self.net.get_position_loggits(act_index, targets_index, target_embs,b.info.mask_targets,position_logits_h, single_batch=False)
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
        act_logits, targets_logits_h,target_embs, position_logits_h, spell_logits, value = self.net(ind,attri_stack,planes_stack,plane_glb)
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
                position_logits = self.net.get_position_loggits(act_id,target_id,target_embs,mask_targets,position_logits_h,single_batch=True)
                position_logits = softmax(position_logits, mask_position, self.device,add_little=True)
                temp_p = self.dist_fn(position_logits)
                if action_known:
                    position_id = action_known[1]
                else:
                    position_id = temp_p.sample()[0].item()
                position_logp = position_logits[0,position_id].item()
            elif act_id == action_type.attack.value:
                mask_targets = env.legal_act(level=1, act_id=act_id)
                targets_logits = self.net.get_target_loggits(act_id, targets_logits_h, single_batch=True)
                targets_logits = softmax(targets_logits, mask_targets, self.device,add_little=True)
                temp_p = self.dist_fn(targets_logits)
                if action_known:
                    target_id = action_known[2]
                else:
                    target_id = temp_p.sample()[0].item()
                target_logp = targets_logits[0,target_id].item()
                if not can_shoot:
                    mask_position = env.legal_act(level=2, act_id=act_id, target_id=target_id)
                    position_logits = self.net.get_position_loggits(act_id, target_id, target_embs, mask_targets,position_logits_h, single_batch=True)
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
                position_logits = self.net.get_position_loggits(act_id, target_id, target_embs, mask_targets,position_logits_h, single_batch=True)
                position_logits = softmax(position_logits, mask_position, self.device,in_train=False)
                position_id = torch.argmax(position_logits,dim=-1)[0].item()
            elif act_id == action_type.attack.value:
                mask_targets = env.legal_act(level=1, act_id=act_id)
                targets_logits = self.net.get_target_loggits(act_id, targets_logits_h, single_batch=True)
                targets_logits = softmax(targets_logits, mask_targets, self.device,in_train=False)
                target_id = torch.argmax(targets_logits, dim=-1)[0].item()
                if not can_shoot:
                    mask_position = env.legal_act(level=2, act_id=act_id, target_id=target_id)
                    position_logits = self.net.get_position_loggits(act_id, target_id, target_embs, mask_targets,position_logits_h, single_batch=True)
                    position_logits = softmax(position_logits, mask_position, self.device,in_train=False)
                    position_id = torch.argmax(position_logits, dim=-1)[0].item()
        return {'act_id': act_id, 'spell_id': spell_id, 'target_id': target_id, 'position_id': position_id,
                'mask_acts': mask_acts, 'mask_spell': mask_spell, 'mask_targets': mask_targets,'mask_position': mask_position,
                'act_logp':act_logp,'target_logp':target_logp,'position_logp':position_logp,'spell_logp':spell_logp,
                'value': value}


    #TODO sil对于behavior target策略差异
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
                act_logits, targets_logits_h,target_embs, position_logits_h, spell_logits, value = self.net(**b.obs)
                act_index = torch.tensor(b.act.act_id, device=self.device, dtype=torch.long).unsqueeze(dim=-1)
                targets_index = torch.tensor(b.act.target_id, device=self.device, dtype=torch.long).unsqueeze(dim=-1)

                '''act gather softmax (batch,act_kinds) by (batch,1) -> (batch,1)'''
                act_logits = softmax(act_logits, mask_acts, self.device,add_little=True)
                prob_act = act_logits.gather(dim=1,index=act_index).squeeze(-1)
                dist_act = self.dist_fn(act_logits)
                # prob_act = dist_act.log_prob(torch.tensor(b.act.act_id, device=self.device)) * torch.tensor(mask_acts, device=self.device)

                '''target'''
                targets_logits = self.net.get_target_loggits(act_index, targets_logits_h, single_batch=False)
                targets_logits = softmax(targets_logits, mask_targets, self.device,add_little=True)
                prob_target = targets_logits.gather(dim=1,index=targets_index).squeeze(-1)
                dist_targets = self.dist_fn(targets_logits)
                # prob_target = dist_targets.log_prob(torch.tensor(b.act.target_id, device=self.device)) * torch.tensor(mask_targets, device=self.device)

                '''position'''
                position_logits = self.net.get_position_loggits(act_index, targets_index, target_embs, mask_targets,position_logits_h, single_batch=False)
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
                surr2 = ratio.clamp(1. - self._eps_clip, 1. + self._eps_clip) * adv
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
                # vf_loss = .5 * (returns - value.squeeze(-1)).pow(2).mean()
                vf_loss = F.smooth_l1_loss(returns,value.squeeze(-1))
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
                nn.utils.clip_grad_norm_(self.net.parameters(),
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
        next_act = BAction.idx_to_action(act_id,position_id,target_id,spell_id,in_battle)
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
        return next_act, {"act":acts,"obs":obs,"mask":mask,"logps":logps,"value":result['value']}


reward_def = 5
r_cum = 0
# def record_sar(buffer,operator,battle,acting_stack,battle_action, obs, acts, mask, logps, value,done,killed_dealt, killed_get,td=True):
#     global r_cum
#     """compute reward"""
#     reward = 0.
#     tmp = 0.
#     bl = len(buffer)
#     if td:
#         #TODO 在overkill条件下是否需要判断win！！！！？？？
#         if battle_action.type == action_type.attack:
#             reward = killed_dealt * battle_action.target.ai_value / battle.ai_value[
#                 ~acting_stack.side] - killed_get * acting_stack.ai_value / battle.ai_value[acting_stack.side]
#             if acting_stack.by_AI != operator:
#                 reward = -reward
#         if done:
#             if battle.by_AI[battle.get_winner()] == operator:  # winner is operator
#                 reward = 1. + reward
#             else:
#                 reward = -1. + reward
#         reward *= reward_def
#         if done or acting_stack.by_AI == operator:
#             tmp= r_cum
#             r_cum = 0.
#         else:
#             r_cum += reward
#         """buffer sar"""
#         if acting_stack.by_AI == operator:
#             if bl:
#                 buffer.rew[bl - 1] += tmp
#             buffer.add(Batch(obs=obs, act=acts, rew=reward, done=done, info=mask, policy={"value": value, "logps": logps}))
#         elif done:
#             if bl:
#                 buffer.rew[bl - 1] += reward
#                 buffer.done[bl - 1] = done
#             else:
#                 logger.info("你的军队还没出手就被干掉了 (╯°Д°)╯︵┻━┻")
#     else:
#         if acting_stack.by_AI == operator:
#             if done:
#                 if acting_stack.side == battle.get_winner():
#                     buffer.add(Batch(obs=obs, act=acts, rew=reward_def, done=True, info=mask, policy={"value": value, "logps": logps}))
#                 else:
#                     buffer.add(Batch(obs=obs, act=acts, rew=-reward_def, done=True, info=mask,policy={"value": value, "logps": logps}))
#             else:
#                 buffer.add(Batch(obs=obs, act=acts, rew=0, done=False, info=mask, policy={"value": value, "logps": logps}))
#         else:
#             if done:
#                 if bl:
#                     if acting_stack.side == battle.get_winner():
#                         buffer.rew[bl - 1] = -reward_def
#                         buffer.done[bl - 1] = True
#                     else:
#                         buffer.rew[bl - 1] = reward_def
#                         buffer.done[bl - 1] = True
#                 else:
#                     logger.info("你的军队还没出手就被干掉了 (╯°Д°)╯︵┻━┻")
# global_buffer = ReplayBuffer(500,ignore_obs_next=True)
# global_buffer_defender = ReplayBuffer(500,ignore_obs_next=True)
# def collect_eps(agent,file=None,battle:Battle=None,n_step = 200,print_act = False,td=False):
#     if not battle:
#         battle = Battle(agent=agent)
#         battle.load_battle(file)
#     battle.checkNewRound()
#     had_acted = False
#     win = False
#     acting_stack = battle.cur_stack
#     if acting_stack.by_AI == 2:
#         battle_action, obs, acts, mask, logps, value = acting_stack.active_stack(ret_obs=True, print_act=print_act)
#     else:
#         battle_action = acting_stack.active_stack()
#         obs, acts, mask, logps, value = None, None, None, None, 0
#     for ii in range(n_step):
#         if acting_stack.by_AI == 2:
#             if not had_acted:
#                 '''by now hand craft AI value is not needed'''
#                 # clear reward cumullated by hand craft AI
#                 # for st in battle.stacks:
#                 #     battle.ai_value[st.side] += st.ai_value * st.amount
#                 had_acted = True
#         damage_dealt, damage_get, killed_dealt, killed_get = battle.doAction(battle_action)
#         battle.checkNewRound()
#         #battle.curStack had updated
#
#         #check done
#         done = battle.check_battle_end()
#         '''single troop get killed over 40% or shooter killed over 10%'''
#         over_killed = False
#         for st in battle.merge_stacks(copy_stack=True):  #FIXME only considered side 0
#             if st.is_shooter and st.amount < st.amount_base * 0.9:
#                 done = True
#                 over_killed = True
#                 break
#             elif st.amount < st.amount_base * 0.6:
#                 done = True
#                 over_killed = True
#                 break
#         #buffer sar
#         if had_acted:
#             record_sar(global_buffer,2,battle,acting_stack,battle_action, obs, acts, mask, logps, value,done,killed_dealt, killed_get,td=td)
#         #next act
#         if done:
#             win = (battle.by_AI[battle.get_winner()] == 2)
#             if over_killed:
#                 win = False
#             break
#         else:
#             acting_stack = battle.cur_stack
#             if acting_stack.by_AI == 2:
#                 print_act = print_act and ii < 5
#                 battle_action, obs, acts, mask, logps, value = acting_stack.active_stack(ret_obs=True,print_act=print_act)
#             else:
#                 battle_action = acting_stack.active_stack()
#                 obs, acts, mask, logps, value = None, None, None, None, 0
#
#     bat,indice = global_buffer.sample(0)
#     global_buffer.reset()
#     return win,bat
class H3SampleCollector:
    def __init__(self,file_list,agent:H3Agent,full_buffer:ReplayBuffer,format_postion=False):
        self.agent = agent
        self.buffer = full_buffer
        self.acted = False
        fl = len(file_list)
        self.defender_states = [] * fl #type:List[BStack]
        file_list_cache = []
        '''cache json'''
        for file in file_list:
            arena = Battle()
            arena.load_battle(file, load_ai_side=False, format_postion=format_postion)
            arena.checkNewRound()
            start = arena.current_state_feature(curriculum=True)
            file_list_cache.append((start, arena.round))
            self.defender_states.append(arena.defender_stacks)
        self.file_list = file_list
        self.file_list_cache = file_list_cache
    def collect_1_ep(self,file=None,battle:Battle=None,n_step = 200,print_act = False,td=False):
        if not battle:
            battle = Battle(agent=self.agent)
            battle.load_battle(file)
        battle.checkNewRound()
        had_acted = False
        win = False
        for ii in range(n_step):
            acting_stack = battle.cur_stack
            if acting_stack.by_AI == 2:
                print_act = print_act and ii < 5
                battle_action, sars = acting_stack.active_stack(ret_obs=True, print_act=print_act)
            else:
                battle_action = acting_stack.active_stack()
                sars = {}  # sars = {"act":None,"obs":None,"mask":None,"logps":None,"value":0}
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

            #check done
            done = battle.check_battle_end()
            '''single troop get killed over 40% or shooter killed over 10%'''
            over_killed = False
            for st in battle.merge_stacks(copy_stack=True):  #FIXME only considered side 0
                if st.is_shooter and st.amount < st.amount_base * 0.9:
                    done = True
                    over_killed = True
                    break
                elif st.amount < st.amount_base * 0.6:
                    done = True
                    over_killed = True
                    break
            #buffer sar
            if had_acted:
                sars["done"] = done
                self.record_sar(2,battle,acting_stack,sars)
            #check done
            if done:
                win = (battle.by_AI[battle.get_winner()] == 2)
                if over_killed:
                    win = False
                break
        return win

    '''[single side Wheel fight]'''
    def collect_eps(self,file_idx,battle:Battle=None,n_step = 200,print_act = False,td=False):
        arena = Battle(by_AI=[2, 1], agent=self.agent)
        obs_idx, obs, defender_stacks = self.choose_start(file_idx)
        arena.load_battle(obs, load_ai_side=False, format_postion=False)
        buf_start = self.buffer._index
        win = False
        for r in range(10):
            last_buf_start = self.buffer._index
            win = self.collect_1_ep(battle=arena)
            if win:
                if r == 0 and obs_idx > 0:
                    '''normal start after middle start
                    need refresh inner states of stacks
                    '''
                    for st in defender_stacks:
                        st.in_battle = arena
                    arena.defender_stacks = defender_stacks
                arena.split_army()
            else:
                break
        '''win all'''
        if win:
            last_buf_start = self.buffer._index
        if r > 0:
            if last_buf_start > buf_start:
                batch_rew = self.buffer.rew[buf_start:last_buf_start]
                self.cumulate_reward(batch_rew)
            else:
                idx = list(range(buf_start, len(self.buffer))) + list(range(last_buf_start))
                batch_rew = self.buffer.rew[idx]
                self.cumulate_reward(batch_rew)
                self.buffer.rew[idx] = batch_rew
        '''win count'''
        return r
    def record_sar(self,operator, battle, acting_stack, sars):
        if acting_stack.by_AI == operator:
            self.acted = True
            obs = sars["obs"]
            acts = sars["act"]
            done = sars["done"]
            mask = sars["mask"]
            logps = sars["logps"]
            value = sars["value"]
            if done:
                if acting_stack.side == battle.get_winner():
                    self.buffer.add(Batch(obs=obs, act=acts, rew=reward_def, done=1, info=mask,policy={"value": value, "logps": logps}))
                else:
                    self.buffer.add(Batch(obs=obs, act=acts, rew=-reward_def, done=1, info=mask,policy={"value": value, "logps": logps}))
            else:
                self.buffer.add(Batch(obs=obs, act=acts, rew=0, done=0, info=mask,policy={"value": value, "logps": logps}))
        else:
            done = sars["done"]
            if done:
                if self.acted:
                    last_idx = self.buffer.last_index
                    if acting_stack.side == battle.get_winner():
                        self.buffer.rew[last_idx] = -reward_def
                        self.buffer.done[last_idx] = 1
                    else:
                        self.buffer.rew[last_idx] = reward_def
                        self.buffer.done[last_idx] = 1
                else:
                    logger.info("你的军队还没出手就被干掉了 (╯°Д°)╯︵┻━┻")
    @staticmethod
    def cumulate_reward(batch):
        a = batch
        s = int(a.sum())
        lk = np.array(range(s - reward_def, -1, -reward_def))
        a[a > 0.5] += lk

    def choose_start(self, idx, from_start=False):
        return -1, self.file_list_cache[idx], None  # RL


class H3SampleCollector_SIL(H3SampleCollector):
    def __init__(self,agent:H3Agent,full_buffer:ReplayBuffer,max_sar_manager):
        super(H3SampleCollector_SIL, self).__init__(agent,full_buffer)
        self.max_sar_manager = max_sar_manager
    def collect_eps(self,file_idx,battle:Battle=None,n_step = 200,print_act = False,td=False):
        arena = Battle(by_AI=[2, 1], agent=self.agent)
        obs_idx, obs, defender_stacks = self.choose_start(file_idx)
        arena.load_battle(obs, load_ai_side=False, format_postion=False)
        buf_start = self.buffer._index
        win = False
        for r in range(10):
            last_buf_start = self.buffer._index
            win = self.collect_1_ep(battle=arena)
            if win:
                if r == 0 and obs_idx > 0:
                    '''normal start after middle start
                    need refresh inner states of stacks
                    '''
                    for st in defender_stacks:
                        st.in_battle = arena
                    arena.defender_stacks = defender_stacks
                arena.split_army()
            else:
                break
        '''win all'''
        if win:
            last_buf_start = self.buffer._index
        if r > 0:
            if last_buf_start > buf_start:
                idx = list(range(buf_start,last_buf_start))
                self.max_sar_manager.update_max(file_idx, obs_idx,self.buffer, sum(self.buffer.rew[buf_start:last_buf_start]),idx )
                batch_rew = self.buffer.rew[buf_start:last_buf_start]
                self.cumulate_reward(batch_rew)
            else:
                idx = list(range(buf_start, len(self.buffer))) + list(range(last_buf_start))
                self.max_sar_manager.update_max(file_idx, obs_idx,self.buffer, sum(self.buffer.rew[idx]), idx)
                batch_rew = self.buffer.rew[idx]
                self.cumulate_reward(batch_rew)
                self.buffer.rew[idx] = batch_rew
        '''win count'''
        return r
    def choose_start(self, idx, from_start=False):
        return self.max_sar_manager.choose_start(idx)
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
    agent.net.device = dev
    agent.net.inpipe.device = dev
def test_game_noGUI(file,agent = None,by_AI = [2,1]):
    #初始化 agent
    test_win = 0
    iter_N = 3
    for ii in range(iter_N):
        battle = Battle(agent=agent)
        battle.load_battle(file)
        battle.checkNewRound()
        # 事件循环(main loop)
        while True:
            next_act = battle.cur_stack.active_stack()
            battle.doAction(next_act)
            battle.checkNewRound()
            if battle.check_battle_end():
                winner = battle.get_winner()
                logger.debug(f"battle end, winner is {winner}")
                if battle.by_AI[winner] == 2:
                    test_win += 1
                break
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
        tgt_stacks = battle.stackQueue
        for i, st in enumerate(tgt_stacks):
            assert st.is_alive(),f"{st} is dead?"
            if st == act.target:
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
    #FIXME fake act at first
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
def dump_episo(ep,dir,file=None):
    if not file:
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
    else:
        dump_in = os.path.join(dir, file)
    np.save(dump_in,ep)
    logger.info(f"episode dumped in {dump_in}")
def load_episo(dir):
    npys = [f for f in os.listdir(dir) if f.endswith(".npy")]
    if len(npys) == 0:
        return
    obss,acts,masks,rews,dones =[],[],[],[],[]
    for f in npys:
        tmp_f = os.path.join(dir,f)
        obs,act, rew, done, mask = np.load(tmp_f, allow_pickle=True)
        obs = Batch.stack(obs)
        act = Batch.stack(act)
        mask = Batch.stack(mask)
        obss.append(obs)
        acts.append(act)
        masks.append(mask)
        rews.append(rew)
        dones.append(done)
    obs2 = Batch.cat(obss)
    act2 = Batch.cat(acts)
    mask2 = Batch.cat(masks)
    rew2 = np.concatenate(rews)
    done2 = np.concatenate(dones)
    '''empty policy will results in batch length = 0'''
    expert = Batch(obs=obs2,obs_next=obs2,act=act2, rew=rew2, done=done2,info=mask2) #,policy = Batch()
    return expert
def start_test():
    import pygame
    pygame.init()  # 初始化pygame
    pygame.display.set_caption('This is my first pyVCMI')  # 设置窗口标题

    actor_critic = H3_net(dev)
    optim = None #torch.optim.Adam(actor_critic.parameters(), lr=0.005)
    dist = torch.distributions.Categorical
    agent = H3Agent(actor_critic, optim, dist, device=dev)
    agent.load_state_dict(torch.load("model_param.pkl"))
    agent.eval()
    agent.in_train = False
    from ENV.H3_battleInterface import start_game_s_gui
    battle = Battle(agent=agent)
    battle.load_battle(file="ENV/battles/0.json")
    start_game_s_gui(battle=battle)


#@profile
def start_train(use_expert_data=False):
    # 初始化 agent
    actor_critic = H3_net(dev)
    lrate = 0.0005
    optim = torch.optim.Adam(actor_critic.parameters(), lr=lrate)
    # actor_critic.act_ids.weight.register_hook(hook_me)
    dist = torch.distributions.Categorical
    agent = H3Agent(actor_critic,optim,dist,device=dev,gae_lambda=0.95)
    # agent.load_state_dict(torch.load("model_param.pkl"))
    count = 0
    # file_list = ['ENV/battles/0.json', 'ENV/battles/1.json', 'ENV/battles/2.json', 'ENV/battles/3.json', 'ENV/battles/4.json']
    file_list = ['ENV/battles/5.json']
    format_postion = False
    replay_buffer = ReplayBuffer(10000, ignore_obs_next=True)
    # max_sar_manager = H3ReplayManager(file_list,agent,use_expert_data=use_expert_data,format_postion = format_postion)
    collector = H3SampleCollector(file_list,agent,replay_buffer)
    cache_idx = range(len(file_list))
    max_win_count = len(file_list)
    if Linux:
        sample_num = 200
    else:
        sample_num = 50
    print_len = 60
    logger.info(f"start training sample eps/epoch = {sample_num}")
    while True:
        agent.eval()
        agent.in_train = True
        collector.buffer.reset()
        for ii in range(sample_num):
            file_idx = random.choice(cache_idx)
            print_act = False
            collector.collect_eps(file_idx)
        batch_data = collector.buffer.sample(0)[0]
        logger.info(len(batch_data.rew))
        agent.train()
        agent.in_train = True
        agent.process_gae(batch_data,single_batch=False)

        amount_idx = np.logical_and(batch_data.obs.attri_stack[:print_len, :, 3] > 1,batch_data.obs.attri_stack[:print_len, :,0] == 0)
        amount_array = batch_data.obs.attri_stack[:print_len, :, 3][amount_idx].reshape(-1, 2).transpose().astype(np.float) / 10
        logger.info(batch_data.done.astype(np.float)[:print_len] * 1.1)
        logger.info("amount")
        logger.info(-batch_data.obs.attri_stack[:print_len, 0, 1].astype(np.float))
        logger.info(amount_array[0])
        logger.info(amount_array[1])
        logger.info("act_logp")
        logger.info(batch_data.policy.logps.act_logp[:print_len])
        logger.info(batch_data.act.act_id.astype(np.float)[:print_len] )
        logger.info("position_logp")
        logger.info(batch_data.policy.logps.position_logp[:print_len])
        logger.info(-(batch_data.act.position_id.astype(np.float)[:print_len] //17) / 10)
        logger.info(-(batch_data.act.position_id.astype(np.float)[:print_len] % 17) / 10 )
        logger.info("target")
        logger.info(batch_data.act.target_id.astype(np.float)[:print_len])
        logger.info("adv")
        logger.info(batch_data.rew.astype(np.float)[:print_len] / 5)
        logger.info(batch_data.adv[:print_len] / 5)
        if Linux:
            to_dev(agent, "cuda")
        loss = agent.learn(batch_data,batch_size=2000)
        # sil_sars = max_sar_manager.get_sars()
        # if len(sil_sars):
        #     sil_sars = Batch.cat(sil_sars)
        #     logger.info("sil act_logp")
        #     logger.info(sil_sars.policy.logps.act_logp[:print_len])
        #     logger.info(sil_sars.act.act_id.astype(np.float)[:print_len])
        #     logger.info("sil adv")
        #     logger.info(sil_sars.rew.astype(np.float)[:print_len] /5)
        #     logger.info(sil_sars.policy.value[:print_len] /5)
        #     loss = agent.learn(sil_sars, batch_size=2000)
        if Linux:
            to_dev(agent, "cpu")
        agent.eval()
        agent.in_train = False
        '''test how many times agent can win'''
        win_count = []
        for file_idx in cache_idx:
            collector.buffer.reset()
            ct = collector.collect_eps(file_idx)
            win_count.append(ct)
            logger.info(f"test-{count}-{file_list[file_idx]} win count = {ct}")
            break
        if sum(win_count) > max_win_count:
            logger.info("model saved")
            torch.save(agent.state_dict(), "model_param.pkl")
            max_win_count = sum(win_count)
        count += 1
        logger.info(f'count={count}')
def start_train_Q():
    # 初始化 ag
    agent = H3AgentQ()
    lrate = 0.0005
    optim = torch.optim.Adam(agent.net.parameters(), lr=lrate)
    agent.optim = optim
    # actor_critic.act_ids.weight.register_hook(hook_me)
    # agent.load_state_dict(torch.load("model_param.pkl"))
    count = 0
    file_list = ['ENV/battles/0.json', 'ENV/battles/1.json', 'ENV/battles/2.json', 'ENV/battles/3.json', 'ENV/battles/4.json']
    # file_list = ['ENV/battles/0.json']
    Q_replay_buffer = ReplayBuffer(200000, ignore_obs_next=True)
    max_sar_manager = H3ReplayManager(file_list,agent)
    collector = H3SampleCollector(agent,Q_replay_buffer,max_sar_manager)
    cache_idx = range(len(file_list))
    max_win_count = len(file_list)
    if Linux:
        sample_num = 100
    else:
        sample_num = 10
    logger.info(f"start training sample eps/epoch = {sample_num}")
    while True:
        agent.eval()
        agent.in_train = True
        for ii in range(sample_num):
            file_idx = random.choice(cache_idx)
            print_act = False
            collector.collect_eps(file_idx)

        batch_data = collector.buffer.sample(0)[0]
        agent.train()
        agent.in_train = True
        # logger.info(batch_data.done.astype(np.float)[:35] * 1.11)
        # logger.info("act_Q")
        # logger.info(batch_data.policy.logps.act_logp[:35])
        # logger.info(batch_data.act.act_id.astype(np.float)[:35] )
        # logger.info("position_logp")
        # logger.info(batch_data.policy.logps.position_logp[:35])
        # logger.info(batch_data.act.position_id.astype(np.float)[:35] //17)
        # logger.info(batch_data.act.position_id.astype(np.float)[:35] % 17 - 9)
        # logger.info("target")
        # logger.info(batch_data.act.target_id.astype(np.float)[:35])
        # logger.info("adv")
        # logger.info(batch_data.adv[:35])
        # logger.info(batch_data.policy.value[:35])
        if Linux:
            to_dev(agent, "cuda")
        loss = agent.learn(batch_data,batch_size=512)
        sil_sars = max_sar_manager.get_sars()
        if len(sil_sars):
            sil_sars = Batch.cat(sil_sars)
            logger.info("sil learning")
            # logger.info(sil_sars.policy.logps.act_logp[:35])
            # logger.info(sil_sars.act.act_id.astype(np.float)[:35])
            # logger.info("act_logp")
            # logger.info(sil_sars.adv[:35])
            # logger.info(sil_sars.policy.value[:35])
            loss = agent.learn(sil_sars, batch_size=512)
        if Linux:
            to_dev(agent, "cpu")
        agent.eval()
        agent.in_train = False


def start_game_record_s():
    arena = Battle(by_AI=[0, 1])
    arena.load_battle("ENV/battles/0.json", load_ai_side=False, format_postion=True)
    arena.split_army()
    arena.checkNewRound()
    data1 = start_game_record(battle=arena)
    #
    arena.split_army()
    arena.checkNewRound()
    data2 = start_game_record(battle=arena)
    data = Batch.cat([data1, data2])
    dump_episo([data.obs, data.act, data.rew, data.done, data.info], "ENV/episode",file='0.npy')
def start_replay_m(file):
    from ENV.H3_battleInterface import start_replay
    data = load_episo(file)
    start_replay(data)
M=0
if __name__ == '__main__':
    if Linux:
        start_train(use_expert_data=False)
    else:
        # start_game_record_s()
        start_train(use_expert_data=False)
        # start_replay_m("ENV/max_sars")  #"ENV/max_sars" episode
        # start_test()

