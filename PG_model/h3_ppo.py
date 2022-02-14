
import sys
from collections import defaultdict
from functools import cmp_to_key

from ding.worker import EpisodeReplayBuffer
from easydict import EasyDict

from ENV.H3_battleInterface import start_game_s_gui

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

np.set_printoptions(precision=1,suppress=True,sign=' ',linewidth=600,formatter={'float': '{: 0.1f}'.format})
logger = get_logger()[1]
logger.setLevel(logging.INFO)
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
                 model,
                 optim: torch.optim.Optimizer=None,
                 dist_fn = torch.distributions.Categorical,
                 device="cpu",
                 discount_factor: float = 0.001,
                 threshold = 0.3,
                 eps_clip: float = .2,
                 vf_coef: float = .5,
                 **kwargs) -> None:
        super().__init__()
        self._eps_clip = eps_clip
        self._w_vf = vf_coef
        self.threshold = threshold
        self.discount_factor = discount_factor
        if not model:
            self.net = H3_Q_net()
        else:
            self.net = model

        self.optim = torch.optim.Adam(model.parameters(), lr=0.02)
        # self._batch_size = 512
        self.dist_fn = dist_fn
        self.device = device
        self.mode = 0 # 0:collect, 1:eval, 2:train
        self.Q1 = defaultdict(defaultdict_int)
        self.Q2 = defaultdict(defaultdict_int)
        self.Q3 = defaultdict(defaultdict_int)
        self.V = defaultdict(lambda: -reward_def - 100.)
        self.sars_count = defaultdict(defaultdict_int)
        self.S = {}
        self.sa1_count = defaultdict(int)
        self.sa2_count = defaultdict(int)
        self.sa3_count = defaultdict(int)
    def Q_to_sars(self):
        sa1sr = []
        for a1,s in self.sa1_count.keys():
            state = self.S[s]
            qa = self.Q1[s][a1]
            # sa1sr.append({'obs': state['obs'],'info': state['info'], 'act': state['act'], 'q': qa})
            sa1sr.append({'obs': state['obs'], 'info': state['info'],'act': state['act'],'mask_a': 1.,'mask_t': 0.,'mask_p': 0.,  'qa': qa,'qt': 0., 'qp': 0.})
        for (a1,a2), s in self.sa2_count.keys():
            state = self.S[s]
            if a1 == 2:
                r'''record move'''
                qp = self.Q2[a1,s][a2]
                sa1sr.append(
                    {'obs': state['obs'], 'info': state['info'],'info': state['info'],'mask_a': 0., 'mask_t': 0., 'mask_p': 1., 'act': state['act'], 'qa': 0.,
                     'qt': 0., 'qp': qp})
            elif a1 == 3:
                r'''record shoot'''
                qt = self.Q2[a1, s][a2]
                sa1sr.append(
                    {'obs': state['obs'], 'info': state['info'],'mask_a': 0., 'mask_t': 1., 'mask_p': 0., 'act': state['act'], 'qa': 0.,
                     'qt': qt, 'qp': 0.})
            else:
                logger.error(f"wrong a1 = {a1}")
                sys.exit(-1)
        for (a1, a2,a3), s in self.sa3_count.keys():
            assert a1 == 3
            r'''only record melee'''
            if a1 == 3 and a3 != 0:
                state = self.S[s]
                qp = self.Q3[(a1, a2),s][a3]
                sa1sr.append(
                    {'obs': state['obs'],'info': state['info'], 'mask_a': 0., 'mask_t': 0., 'mask_p': 1., 'act': state['act'], 'qa': 0.,
                     'qt': 0., 'qp': qp})
        # sars = []
        # for a,s in self.sars_count.keys():
        #     a1,a2,a3 = a
        #     qa = self.Q1[s][a1]
        #     if a1 < 2:
        #         qt = qp = 0
        #     elif a1 == 2:
        #         qp = self.Q2[a1,s][a2]
        #         qt = 0
        #     elif a1 == 3:
        #         qt = self.Q2[a1,s][a2]
        #         qp = self.Q3[(a1, a2),s][a3]
        #     else:
        #         logger.error(f'wrong act = {a1}')
        #         sys.exit(-1)
        #     state = self.S[s]
        #     sars.append({'obs': state['obs'],'info': state['info'], 'act': state['act'], 'q1': qa,'q2': qt,'q3': qp})  # 'next_obs':torch.tensor(s_)
        sars = Batch.stack(sa1sr)
        return sars
    def clear_table(self):
        self.V.clear()
        self.S.clear()
        self.Q1.clear()
        self.Q2.clear()
        self.Q3.clear()
        self.sa1_count.clear()
        self.sa2_count.clear()
        self.sa3_count.clear()
        self.sars_count.clear()
    def pruning(self):
        self.clear_table()
        self.process_max_tree(self)
    def process_max_tree(self,batch):
        for i, traj in enumerate(batch):
            for j,t in enumerate(reversed(traj)):
                s = tuple(map(tuple, t['obs']['attri_stack']))
                a = (t['act']['act_id'],t['act']['position_id'],t['act']['target_id'])  #{'act_id': act_id, 'spell_id': spell_id, 'target_id': target_id, 'position_id': position_id,
                if a[0] == 3:
                    a = (a[0],a[2],a[1])
                s_ = tuple(map(tuple, t['obs_next']['attri_stack']))
                if s not in self.S:
                    self.S[s] = {'obs':t['obs'],'info':t['info'],'act':t['act']}
                r'''                                           
                        G1   G2   G3
                          r12  r23  r3T                                          
                        S1 - S2 - S3 - T   
                        Gs对应单次episode中的特定s,eg. s1-s2-s3-s1-s3-T
                        对于every visit MC来说,episode中两次Gs1并不相同  Vs1=E(Gs1)
                        对于first visit MC,则相同,Vs1=max(Gs1) 
                        -> Gs3 = Vs3 = max(Vs3,max(Qs3a)) step1
                        -> Gs2 = γ* Gs3+rs2s3            step2
                        -> Qs2a = Esa(Gs2)                step3
                                -> Gs2 = Vs2 = max(Vs2,max(Qs2a))   step1
                                -> Gs1 = γ* Gs2+rs1s2              step2
                                -> Qs1a = Esa(Gs1)                  step3
                '''
                if t['done']:
                    first_done = True
                    T = (0,0,0,a,s)
                    n = self.sars_count[a,s][T]
                    n += 1
                    self.sars_count[a,s][T] = n
                    # Gt = 0.
                    r = t['rew']
                    r'''step1 Vt = E(r+γ* Gt)= E(r) 因为Gt=0'''
                    self.V[T] += (r - self.V[T]) / n
                else:
                    if not first_done:
                        continue
                    self.sars_count[a,s][s_] += 1
                    vs_ = max(self.Q1[s_].values())
                    r'''step1 Vs_ = max(Vs_,max(Qs_a))'''
                    self.V[s_] = Gs_ = max([self.V[s_], vs_])
                    r'''
                   pre_step2  预备工作
                   Gs = rss' + Gs_ - self.discount_factor
                   为保证rss'不随策略pi改变,一种最简单实现方式为中间reward统统=0
                   '''
                    r = t['rew']
                    self.V[s_] += r
                a1, a2, a3 = a
                self.sa1_count[a1, s] += 1
                r'''step3                                            step2  V[s_] == Gs_, 其他s_下的r=0   '''
                qvalue = sum([self.sars_count[a, s][s_] * (self.V[s_] - self.discount_factor) for s_ in self.sars_count[a, s].keys()]) / \
                sum(self.sars_count[a, s].values())
                if a1 < 2:
                    self.Q1[s][a1] = qvalue
                elif a1 == 2:
                    self.Q2[a1, s][a2] = qvalue
                    self.Q1[s][a1] = max(self.Q2[a1,s].values())
                    self.sa2_count[(a1,a2),s] += 1
                elif a1 == 3:
                    self.Q3[(a1,a2),s][a3] = qvalue
                    self.Q2[a1,s][a2] = max(self.Q3[(a1,a2),s].values())
                    self.Q1[s][a1] = max(self.Q2[a1,s].values())
                    self.sa2_count[(a1,a2),s] += 1
                    self.sa3_count[(a1,a2,a3), s] += 1
                else:
                    logger.error(f'wrong act = {a1}')
                    sys.exit(-1)
                r'''terminal步的Vs_没有意义'''
                if not t['done']:
                    self.V[s_] -= r
            vs = max(self.Q1[s].values())
            r'''step1 Vs_ = max(Vs_,max(Qs_a))'''
            self.V[s] = max([self.V[s], vs])
            assert self.V[s] > -2.
        return self.V[s]
    def show_traj_Gs(self,batch):
        for i, traj in enumerate(batch):
            traj.Gs = np.zeros((len(traj.rew),),dtype=np.float32)
            traj.q1 = np.zeros((len(traj.rew),), dtype=np.float32)
            traj.q2 = np.zeros((len(traj.rew),), dtype=np.float32)
            traj.q3 = np.zeros((len(traj.rew),), dtype=np.float32)
            mask_targets = traj.info.mask_targets
            act_index = torch.tensor(traj.act.act_id, device=self.device, dtype=torch.long).unsqueeze(
                dim=-1)
            targets_index = torch.tensor(traj.act.target_id, device=self.device,
                                         dtype=torch.long).unsqueeze(dim=-1)
            position_index = torch.tensor(traj.act.position_id, device=self.device,
                                          dtype=torch.long).unsqueeze(dim=-1)

            current_act_q, Va, Va_mask, targets_logits_h, target_embs, position_logits_h = self.net(single_batch=False,
                                                                                                    **traj.obs)
            current_targets_q, Vt, Vt_mask = self.net.get_target_q(act_index, targets_logits_h, single_batch=False)
            current_position_q, Vp, Vp_mask = self.net.get_position_q(act_index, targets_index, target_embs,
                                                                      mask_targets, position_logits_h,
                                                                      single_batch=False)
            qa_value = current_act_q.gather(1, act_index).squeeze(-1).detach().numpy()
            qa_softm = current_act_q.softmax(dim=-1).gather(1, act_index).squeeze(-1).detach().numpy()
            qt_value = current_targets_q.gather(1, targets_index).squeeze(-1).detach().numpy()
            qt_softm = current_targets_q.softmax(dim=-1).gather(1, targets_index).squeeze(-1).detach().numpy()
            qp_value = current_position_q.gather(1, position_index).squeeze(-1).detach().numpy()
            qp_softm = current_position_q.softmax(dim=-1).gather(1, position_index).squeeze(-1).detach().numpy()
            traj.q3_value = qp_value
            traj.q2_value = qt_value
            traj.q1_value = qa_value
            for j, t in enumerate(traj):
                s = tuple(map(tuple, t['obs']['attri_stack']))
                a = (t['act']['act_id'], t['act']['position_id'], t['act'][
                    'target_id'])  # {'act_id': act_id, 'spell_id': spell_id, 'target_id': target_id, 'position_id': position_id,
                a1, a2, a3 = a
                if a[0] == 3:
                    a1,a2,a3 = a[0], a[2], a[1]
                s_ = tuple(map(tuple, t['obs_next']['attri_stack']))
                vs = self.V[s]
                assert vs > -2
                traj.Gs[j] = vs
                traj.q1[j] = self.Q1[s][a1]
                traj.q2[j] = self.Q2[a1,s][a2]
                traj.q3[j] = self.Q3[(a1,a2), s][a3]
        print()
    def choose_act(self,act_logits,mask_acts):
        # Use large negative number to mask actions from argmax
        if self.mode == 0:
            act_id = self.dist_fn((act_logits - (1. - mask_acts) * 1e8).softmax(dim=-1)).sample()[0].item()
        elif self.mode == 1:
            act_id = int((act_logits - (1. - mask_acts) * 1e8).argmax(1))
        else:
            print("batch mode should not be here")
            exit(-1)
        return act_id
    def choose_target(self,act_id, targets_logits_h,mask_targets):
        targets_logits = self.net.get_target_q(act_id, targets_logits_h, single_batch=True)
        # Use large negative number to mask actions from argmax
        if self.mode == 0:
            target_id = self.dist_fn((targets_logits - (1. - mask_targets) * 1e8).softmax(dim=-1)).sample()[0].item()
        elif self.mode == 1:
            target_id = int((targets_logits - (1. - mask_targets) * 1e8).argmax(1))
        else:
            print("batch mode should not be here")
            exit(-1)
        return target_id
    def choose_position(self,act_id, target_id, target_embs, mask_targets,position_logits_h,mask_position):
        '''target id is -1 mask_targets = all 0'''
        position_logits = self.net.get_position_q(act_id, target_id, target_embs, mask_targets,
                                                                 position_logits_h, single_batch=True)
        if self.mode == 0:
            position_id = self.dist_fn((position_logits - (1. - mask_position) * 1e8).softmax(dim=-1)).sample()[0].item()
        elif self.mode == 1:
            position_id = int((position_logits - (1. - mask_position) * 1e8).argmax(1))
        else:
            print("batch mode should not be here")
            exit(-1)
        assert mask_position[position_id]
        return position_id
    #单次输入 obs->act,mask
    def forward(self, ind,attri_stack,planes_stack,plane_glb,env,can_shoot = False,print_act = False) -> Batch:
        act_logits, targets_logits_h,target_embs, position_logits_h = self.net(ind,attri_stack,planes_stack,plane_glb,single_batch=True)
        '''ind,attri_stack,planes_stack,plane_glb shape like (batch,...) or (1,...) when inference'''
        #TODO 把act target position推理 并行化
        assert self.mode < 2
        mask_acts = env.legal_act(level=0)
        act_id = self.choose_act(act_logits,mask_acts)
        if act_id == action_type.move.value:
            '''no target'''
            mask_targets = np.zeros(14, dtype=bool)
            target_id = -1
            '''position only'''
            mask_position = env.legal_act(level=1, act_id=act_id)
            position_id = self.choose_position(act_id, target_id, target_embs, mask_targets,position_logits_h,mask_position)
        elif act_id == action_type.attack.value:
            mask_targets = env.legal_act(level=1, act_id=act_id)
            target_id = self.choose_target(act_id, targets_logits_h, mask_targets)
            if can_shoot:
                '''no position'''
                mask_position = np.zeros(11 * 17, dtype=bool)
                position_id  = -1
            else:
                mask_position = env.legal_act(level=2, act_id=act_id, target_id=target_id)
                position_id = self.choose_position(act_id, target_id, target_embs, mask_targets, position_logits_h,mask_position)
        else:
            '''no target'''
            mask_targets = np.zeros(14, dtype=bool)
            target_id = -1
            '''no position'''
            mask_position = np.zeros(11 * 17, dtype=bool)
            position_id = -1
        return {'act_id': act_id, 'target_id': target_id, 'position_id': position_id,
                'mask_acts': mask_acts, 'mask_targets': mask_targets,'mask_position': mask_position}

    #@profile
    #批量输入训练
    def learn(self, batch, batch_size=None, repeat=4, **kwargs):
        pcount = 0
        for _ in range(repeat):
            for b in batch.split(batch_size,shuffle=False):
                # mask_act = b.info.
                mask_targets = b.info.mask_targets
                # mask_position = b.info.mask_position
                act_index = torch.tensor(b.act.act_id, device=self.device, dtype=torch.long).unsqueeze(dim=-1)
                targets_index = torch.tensor(b.act.target_id, device=self.device, dtype=torch.long).unsqueeze(dim=-1)
                position_index = torch.tensor(b.act.position_id, device=self.device,
                                              dtype=torch.long).unsqueeze(dim=-1)
                current_act_q, Va,Va_mask ,targets_logits_h, target_embs, position_logits_h = self.net(single_batch=False,**b.obs)
                current_targets_q, Vt,Vt_mask = self.net.get_target_q(act_index, targets_logits_h,single_batch=False)
                current_position_q, Vp,Vp_mask = self.net.get_position_q(act_index, targets_index, target_embs, mask_targets,position_logits_h,single_batch=False)
                # logit, action, q_value, adv, return_, weight = data
                qa = torch.tensor(b.qa, device=self.device, dtype=torch.float32)
                qa_value = current_act_q.gather(1, act_index).squeeze(-1)
                qa_v_mask = torch.tensor(b.mask_a, device=self.device, dtype=torch.float32)
                qt = torch.tensor(b.qt, device=self.device, dtype=torch.float32)
                qt_value = current_targets_q.gather(1, targets_index).squeeze(-1)
                qt_v_mask = torch.tensor(b.mask_t, device=self.device, dtype=torch.float32)
                qp = torch.tensor(b.qp, device=self.device, dtype=torch.float32)
                qp_value = current_position_q.gather(1, position_index).squeeze(-1)
                qp_v_mask = torch.tensor(b.mask_p, device=self.device, dtype=torch.float32)
                qa_loss = F.smooth_l1_loss(qa, qa_value*qa_v_mask, reduction='sum')  # .sum()
                qt_loss = F.smooth_l1_loss(qt, qt_value*qt_v_mask, reduction='sum')  # .sum()
                qp_loss = F.smooth_l1_loss(qp, qp_value*qp_v_mask, reduction='sum')  # .sum()
                # entropy_loss = (dist.entropy()).sum()
                loss = qa_loss + qt_loss + qp_loss
                # if pcount < 5:
                #     logger.info("clip_loss={:.4f} vf_loss={:.4f} e_loss={:.4f}".format(clip_loss,vf_loss,e_loss))
                pcount += 1
                # losses.append(loss.item())
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
    def choose_action(self,in_battle:Battle,ret_obs = False,print_act=False,action_known:tuple=None):
        ind, attri_stack, planes_stack, plane_glb = in_battle.current_state_feature(curriculum=False)
        with torch.no_grad():
            result = self.forward(ind[None], attri_stack[None], planes_stack[None], plane_glb[None], in_battle,
                           can_shoot=in_battle.cur_stack.can_shoot(), print_act=print_act)
        act_id = result['act_id']
        position_id = result['position_id']
        target_id = result['target_id']
        # spell_id = result['spell_id']
        next_act = BAction.idx_to_action(act_id,position_id,target_id,in_battle)
        if not ret_obs:
            return next_act
        act_id = 0 if act_id < 0 else act_id
        position_id = 0 if position_id < 0 else position_id
        target_id = 0 if target_id < 0 else target_id
        # spell_id = 0 if spell_id < 0 else spell_id
        obs = {'ind': ind, 'attri_stack': attri_stack, 'planes_stack': planes_stack, 'plane_glb': plane_glb}
        acts = {'act_id': act_id, 'position_id': position_id, 'target_id': target_id}
        mask = {'mask_acts': result['mask_acts'], 'mask_targets': result['mask_targets'], 'mask_position': result['mask_position']}
        return next_act,{"act":acts,"obs":obs,"mask":mask}
    def get_end_Q(self,sars):
        s = tuple(map(tuple, sars.obs.attri_stack))
        a1, a2, a3 = (sars.act.act_id, sars.act.position_id, sars.act.target_id)
        assert a1 < 4
        if a1 < 2:
            q = self.Q1[s][a1]
        elif a1 == 2:
            q = self.Q2[a1, s][a2]
        else:
            a2, a3 = a3, a2
            q = self.Q3[(a1, a2), s][a3]
        return q

    def cmp_single_state(self,sars1,sars2):
        q1 = self.get_end_Q(sars1)
        q2 = self.get_end_Q(sars2)
        error1 = q1 - q2
        if abs(error1) < 1E-6:
            error1 = 0
        return error1
    def cmp_reward_func(self,episode1,episode2):
        assert episode1.done[-1]
        assert episode2.done[-1]
        # r1 = r2 = 0
        # for ii,episode in enumerate([episode1,episode2]):
        #     rr = sum(episode.rew)
        #     if episode.rew[-1] < 0 and sum(episode.done) > 1:
        #         rr += 1.
        #     # rr = rr - len(episode.rew) * self.discount_factor
        #     if ii == 0:
        #         r1 = rr
        #     else:
        #         r2 = rr
        length = min(len(episode1.rew),len(episode2.rew))
        for i in range(length):
            err = self.cmp_single_state(episode1[i],episode2[i])
            if abs(err) > 1E-6:
                return err
        return 0
    def start_train(self):
        # 初始化 ag
        agent = self
        # actor_critic.act_ids.weight.register_hook(hook_me)
        # agent.load_state_dict(torch.load("model_param.pkl"))
        count = 0
        # file_list = ['ENV/battles/0.json', 'ENV/battles/1.json', 'ENV/battles/2.json', 'ENV/battles/3.json',
        #              'ENV/battles/4.json']
        file_list = ['ENV/battles/0.json']
        Q_replay_buffer = ReplayBuffer(200000, ignore_obs_next=True)
        # max_sar_manager = H3ReplayManager(file_list, agent)
        collector = H3SampleCollector(record_sar_max_tree,file_list,agent, Q_replay_buffer)
        expert_list = load_episo("ENV/episode")
        for expert in expert_list:
            idx = np.argwhere(expert.done)[:-1]
            expert.obs_next[idx] = expert.obs[idx + 1]
            expert.policy = Batch()
            expert.policy.done_ = expert.done
            expert.done[:-1] = 0
            attri_stack_orig = expert.obs['attri_stack']
            expert.policy.attri_stack_orig = attri_stack_orig
            expert.obs['attri_stack'] = get_tuple(attri_stack_orig)
            attri_stack_orig = expert.obs_next['attri_stack']
            # expert.policy.attri_stack_orig = attri_stack_orig
            expert.obs_next['attri_stack'] = get_tuple(attri_stack_orig)
            collector.best_hist_buffer.append(expert)
        cache_idx = range(len(file_list))
        if Linux:
            sample_num = 500
        else:
            sample_num = 100
        logger.info(f"start training sample eps/epoch = {sample_num}")
        print_len_ = 35
        eval_frq = 5
        bi = None
        while True:
            logger.info(f'train iter = {count}')
            agent.eval()
            agent.mode = 0
            for ii in range(sample_num):
                file_idx = random.choice(cache_idx)
                print_act = False
                collector.collect_eps(file_idx)

            current_data = collector.current_buffer
            history = collector.history_buffer.sample(400,0)
            batch_data = current_data
            if history:
                batch_data = batch_data + history
            if len(collector.best_hist_buffer):
                batch_data = batch_data + collector.best_hist_buffer
            agent.train()
            agent.mode = 2
            self.clear_table()
            r'''build max tree'''
            self.Vstart = self.process_max_tree(batch_data)
            r'''[print(b.act.act_id) for b in batch_data]
             min(self.V.values())
             '''
            r'''剪枝'''
            batch_data.sort(key=cmp_to_key(self.cmp_reward_func),reverse=True)
            collector.best_hist_buffer = batch_data[:50]
            self.clear_table()
            collector.history_buffer.push(current_data,-1)
            collector.current_buffer.clear()
            batch_data_treed = self.Q_to_sars()
            if True: # debug log
                logger.info(f'batch data size = {len(batch_data)}')
                logger.info(f'batch tree size = {len(batch_data_treed)}')
                best_hist_buffer_1 = collector.best_hist_buffer[0]
                print_len = min(len(best_hist_buffer_1),print_len_)
                me_amount = np.zeros((print_len, 2))
                enemy_amount = np.zeros((print_len, 3))
                if (count + 1) % eval_frq == 0:
                    self.show_traj_Gs(collector.best_hist_buffer[:5])

                    for ii in range(5):
                        bf = collector.best_hist_buffer[ii]
                        logger.info('Gs')
                        logger.info(bf.Gs)
                        logger.info('q - qvalue')
                        logger.info(bf.q1)
                        logger.info(bf.q1_value)
                        logger.info(bf.q2)
                        logger.info(bf.q2_value)
                        logger.info(bf.q3)
                        logger.info(bf.q3_value)
                        # bf.obs.attri_stack = bf.policy.attri_stack_orig
                        # dump_episo([bf.obs, bf.act, bf.rew, bf.done, bf.info], "ENV/max_sars",
                        #            f'max_data_{ii}.npy')
                        # logger.info(f"states dumped in max_data_{ii}.npy")
                mask_acts = torch.tensor(best_hist_buffer_1.info.mask_acts,dtype=torch.float32)
                mask_targets = best_hist_buffer_1.info.mask_targets
                act_index = torch.tensor(best_hist_buffer_1.act.act_id, device=self.device, dtype=torch.long).unsqueeze(dim=-1)
                targets_index = torch.tensor(best_hist_buffer_1.act.target_id, device=self.device, dtype=torch.long).unsqueeze(dim=-1)
                position_index = torch.tensor(best_hist_buffer_1.act.position_id, device=self.device,
                                             dtype=torch.long).unsqueeze(dim=-1)

                current_act_q, Va, Va_mask, targets_logits_h, target_embs, position_logits_h = self.net(single_batch=False,
                                                                                                        **best_hist_buffer_1.obs)
                # current_targets_q, Vt, Vt_mask = self.net.get_target_q(act_index, targets_logits_h, single_batch=False)
                current_position_q, Vp, Vp_mask = self.net.get_position_q(act_index, targets_index, target_embs,
                                                                          mask_targets, position_logits_h,
                                                                          single_batch=False)
                qa_value = current_act_q.detach().numpy()
                qa_softm = (current_act_q - (1-mask_acts)*1E8).softmax(dim=-1)
                qa_softm_act = qa_softm.gather(1, act_index).squeeze(-1).detach().numpy()
                qa_softm_max = qa_softm.argmax(-1).float().squeeze(-1).detach().numpy()
                # qt_value = current_targets_q.gather(1, targets_index).squeeze(-1).detach().numpy()
                # qt_softm = current_targets_q.softmax(dim=-1).gather(1, targets_index).squeeze(-1).detach().numpy()
                qp_value = current_position_q.gather(1,position_index ).squeeze(-1).detach().numpy()
                qp_softm = current_position_q.softmax(dim=-1).gather(1, position_index).squeeze(-1).detach().numpy()
                for i in range(print_len):
                    stacks = best_hist_buffer_1.obs.attri_stack[i]
                    mask = np.logical_and(stacks[:, 0] == 0, stacks[:, 3] > 1)
                    me_am = stacks[mask][:, 3]
                    me_amount[i, :min(2, len(me_am))] = me_am[:2]

                    mask = stacks[:, 0] == 1
                    enemy_am = stacks[mask][:, 3]
                    enemy_amount[i, :min(3, len(enemy_am))] = enemy_am[:3]

                me_amount = me_amount.transpose().astype(np.float) / 10
                enemy_amount = enemy_amount.transpose().astype(np.float) / 10
                # logger.info(best_hist_buffer_1.done.astype(np.float)[:print_len] * 1.1)
                # logger.info("amount")
                # logger.info(-best_hist_buffer_1.obs.attri_stack[:print_len, 0, 1].astype(np.float))
                # logger.info(me_amount[0])
                # logger.info(me_amount[1])
                # logger.info("enemy_amount")
                # logger.info(enemy_amount[0])
                # logger.info(enemy_amount[1])
                # logger.info(enemy_amount[2])
                logger.info("act_qvalue 1 - 5")
                logger.info(qa_softm_act[:print_len])
                logger.info(qa_value[:print_len,0])
                logger.info(qa_value[:print_len,1])
                logger.info(qa_value[:print_len,2])
                logger.info(qa_value[:print_len,3])
                logger.info(qa_value[:print_len,4])
                logger.info(qa_softm_max[:print_len])
                logger.info(best_hist_buffer_1.act.act_id.astype(np.float)[:print_len])
                logger.info("position_q")
                logger.info(qp_value[:print_len])
                logger.info(qp_softm[:print_len])
                logger.info(-(best_hist_buffer_1.act.position_id.astype(np.float)[:print_len] // 17) / 10)
                logger.info(-(best_hist_buffer_1.act.position_id.astype(np.float)[:print_len] % 17) / 10)
                logger.info("target")
                logger.info(best_hist_buffer_1.act.target_id.astype(np.float)[:print_len])
                logger.info(f"G = {self.Vstart}")

                # logger.info(best_hist_buffer_1.reward_cum.astype(np.float)[:print_len])
                # logger.info(best_hist_buffer_1.Gs.astype(np.float)[:print_len])
                r'''
                from operator import itemgetter
                keys = list(self.Q1.keys())[:10]
                out = itemgetter(*keys)(self.Q1)
                '''
            if Linux:
                self.to_dev(agent, "cuda")
            loss = agent.learn(batch_data_treed,repeat=10, batch_size=5120)
            if Linux:
                self.to_dev(agent, "cpu")
            agent.eval()
            agent.mode = 1
            observe = False #((count + 1) % eval_frq == 0)
            if observe:
                arena = Battle(by_AI=[2, 1], agent=self)
                arena.load_battle(file=file_list[0])

                bi = start_game_s_gui(battle=arena,battle_int=bi)
            else:
                '''test how many times agent can win'''
                wr, wc = collector.get_win_rates()
                logger.info(f'win rate = {wr}')
                logger.info(f'cont = {wc}')
                test_battle_idx = cache_idx
                win_count = []
                collector.reset_counts()
                for file_idx in test_battle_idx:
                    collector.buffer.reset()
                    ct = collector.collect_eps(file_idx, from_start=True)
                    win_count.append(ct)
                    logger.info(f"test-{count}-{file_list[file_idx]} win count = {ct}")
                if sum(win_count) > collector.max_win_count:
                    logger.info("model saved")
                    torch.save(self.state_dict(), "model_param.pkl")
                    collector.max_win_count = sum(win_count)
            count += 1
            logger.info(f'count={count}')
class H3ReplayManager_interface:
    def init_expert_data(self):
        raise Exception('interface should be implemented')
    def choose_start(self,idx,from_start=False):
        raise Exception('interface should be implemented')
    def update_max(self, idx, obs_idx, data: ReplayBuffer, start, end):
        raise Exception('interface should be implemented')
    def get_sars(self):
        raise Exception('interface should be implemented')
class H3ReplayManager(H3ReplayManager_interface):
    def __init__(self,file_list,agent,p_start_from_scratch = 0.5,use_expert_data=False, format_postion=False):
        fl = len(file_list)
        self.defender_states = []  # type:List[BStack]
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
        self.max_sar = [None] * fl #type:List[Batch]
        self.agent = agent
        self.use_expert_data = use_expert_data
        if use_expert_data:
            self.init_expert_data()
        self.p_start_from_scratch = p_start_from_scratch
    def init_expert_data(self):
        expert = load_episo("ENV/episode")
        if expert:
            compute_reward_from_episodes(expert,0,len(expert))
            self.max_sar[0] = Batch(expert,copy=True)
            self.agent.process_gae(expert, single_batch=False, sil=True)
            '''fill in policy'''
            self.max_sar[0].policy = expert.policy
    def choose_start(self,idx,from_start=False):
        if not self.max_sar[idx] or from_start or np.random.binomial(1, self.p_start_from_scratch):
            return -1, self.file_list_cache[idx], None  # RL
        else:
            obs_idx = random.choice(range(len(self.max_sar[idx].obs)))  # SIL
            return obs_idx, (self.max_sar[idx].obs[obs_idx].attri_stack, 0), [copy.copy(st) for st in self.defender_states[idx]]  # FIXME round = 0
    def update_max(self,idx,obs_idx,data:ReplayBuffer,start,end):
        assert end != start
        assert data.done[end - 1] > 0
        if end < start:
            logger.info('buff size is not enough update_max return')
            return
        if data.rew[end - 1] < 0:
            tmp_data = data[start:end]
            record_done = int(sum(tmp_data.done))
            if record_done < 2:
                return
            br_index = np.where(tmp_data.rew > 0)
            end_bias = br_index[-1][-1] + 1
            end = start + end_bias
        data_rew = data.rew[start:end]
        data_rew = sum(data_rew)
        if not self.max_sar[idx]:
            if data_rew > 0:
                self.max_sar[idx] = data[start:end]
                max_data = self.max_sar[idx]
                dump_episo([max_data.obs, max_data.act, max_data.rew, max_data.done, max_data.info], "ENV/max_sars",
                           f'max_data_{idx}.npy')
                logger.info(f"states dumped in max_data_{idx}.npy")
        else:
            record_done = int(sum(self.max_sar[idx].done))
            data_done = int(sum(data.done[start:end]))
            record_rew = sum(self.max_sar[idx].rew)
            if record_rew < data_rew or record_done < data_done:
                if obs_idx <= 0:
                    logger.info(f"idx-{idx} get new traj from scratch")
                else:
                    logger.info(f"idx-{idx} get new traj")
                self.max_sar[idx] = data[start:end]
                max_data = self.max_sar[idx]
                dump_episo([max_data.obs, max_data.act, max_data.rew, max_data.done, max_data.info], "ENV/max_sars",f'max_data_{idx}.npy')
                logger.info(f"states dumped in max_data_{idx}.npy")
    def get_sars(self):
        bats = []
        for exp in self.max_sar:
            if exp:
                exp_copy = Batch(exp,copy=True)
                cumulate_reward(exp_copy,0,len(exp_copy))
                self.agent.process_gae(exp_copy, single_batch=False, sil=True)
                bats.append(exp_copy)
                logger.info(f"exp.rew={sum(exp.rew)}")
        return bats
class H3ReplayManager_SIL(H3ReplayManager):
    def __init__(self,file_list,agent,p_start_from_scratch = 0.5,use_expert_data=False,format_postion=False):
        super(H3ReplayManager_SIL, self).__init__(file_list,agent,p_start_from_scratch,use_expert_data,format_postion)
    def choose_start(self,idx=-1,from_start=False):
        if idx < 0:
            idx = random.choice(range(len(self.max_sar)))
        obs_idx, obs, defs = super(H3ReplayManager_SIL, self).choose_start(idx,from_start=from_start)
        return idx, obs_idx, obs, defs
class H3ReplayManager_SIL_tree(H3ReplayManager_interface):
    def __init__(self,file_list,agent,p_start_from_scratch = 0.5,use_expert_data=False, format_postion=False):
        fl = len(file_list)
        self.defender_states = []  # type:List[BStack]
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
        self.max_sar = [None] * fl #type:List[StateNode]
        self.agent = agent
        self.use_expert_data = use_expert_data
        if use_expert_data:
            self.init_expert_data()
        self.p_start_from_scratch = p_start_from_scratch
    def init_expert_data(self):
        expert = load_episo("ENV/episode")
        if expert:
            compute_reward_from_episodes(expert, 0, len(expert))
            self.max_sar[0] = Batch(expert, copy=True)
            self.agent.process_gae(expert, single_batch=False, sil=True)
            '''fill in policy'''
            self.max_sar[0].policy = expert.policy
    # def build_tree_node(self,state_node:StateNode,sar):
    #     obs,acts,mask = sar
    #     if not state_node:
    #         root = StateNode(None,obs,mask.act_mask)
    #         act_id = acts.act_id
    #         target_id = acts.target_id
    #         position_id = acts.position_id
    #         if act_id == action_type.wait.value:
    #             return ActionNode(root,act_id,None)
    #         elif act_id == action_type.defend.value:
    #             return ActionNode(root,act_id,None)
    #         elif act_id == action_type.move.value:
    #             move = ActionNode(root,act_id,None)
    #             position = ActionNode(move,position_id,None)



def get_tuple(attri_stack_orig):
    # attri_stack[i] = np.array(
    #     [st.side, st.slotId, st.id, st.amount, st.amount_base, st.first_HP_Left, st.health, st.luck, st.attack,
    #      st.defense, st.max_damage, st.min_damage,
    #      st.speed, st.morale, st.shots, st.y, st.x, int(st.had_moved), int(st.had_waited), int(st.had_retaliated),
    #      int(st.had_defended)])
    # mask = attri_stack_orig[...,9] > 10
    # if mask.any():
    #     print()
    attri_stack = np.copy(attri_stack_orig)
    r'''部队基础总生命值'''
    health_baseline = attri_stack[..., 4] * attri_stack[..., 6]
    health_baseline = health_baseline.astype('int')
    r'''部队基础总生命值中最大值'''
    health_baseline_max = np.max(health_baseline,axis=-1)
    r'''部队当前总生命值'''
    health_current = np.clip(attri_stack[..., 3] - 1, 0, np.inf) * attri_stack[..., 6] + attri_stack[..., 5]
    health_current = health_current.astype('int')
    r'''部队基础总生命值 / 部队基础总生命值中最大值'''
    health_ratio_bymax = health_baseline * 10 // health_baseline_max[...,None]
    r'''部队当前总生命值 / 部队基础总生命值'''
    health_current_ratio = (health_current * 10 // (health_baseline + 1E-9)).astype('int') + (
            health_current > 0).astype('int')
    r'''部队当前数量 / 部队基础数量'''
    amount_ratio = (attri_stack[..., 3] * 10 // (attri_stack[..., 4] + 1E-9)).astype('int') + (
                attri_stack[..., 3] > 0).astype('int')
    r'''远程弹药数量 / 基数16'''
    shoots_ratio = (attri_stack[..., 14] * 4 // 16.0001).astype('int') + (attri_stack[..., 14] > 0).astype('int')
    attri_stack[..., 3] = amount_ratio
    attri_stack[..., 4] = 0
    attri_stack[..., 5] = health_current_ratio
    attri_stack[..., 6] = health_ratio_bymax
    attri_stack[..., 14] = shoots_ratio


    # attri_stack = tuple(map(tuple, attri_stack))
    return attri_stack
def defaultdict_int():
    return defaultdict(int)
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
                 ent_coef: float = .1,
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
                # spell_logits = softmax(spell_logits, mask_spell, self.device,add_little=True)
                # prob_spell = spell_logits.gather(dim=1,index=torch.tensor(b.act.spell_id, device=self.device,dtype=torch.long).unsqueeze(dim=-1)).squeeze(-1)
                # dist_spell = self.dist_fn(spell_logits)
                # prob_spell = dist_spell.log_prob(torch.tensor(b.act.spell_id, device=self.device)) * torch.tensor(mask_spell, device=self.device)

                old_prob_act = torch.tensor(b.policy.logps.act_logp, device=self.device)
                old_prob_target = torch.tensor(b.policy.logps.target_logp, device=self.device)
                old_prob_position = torch.tensor(b.policy.logps.position_logp, device=self.device)
                # old_prob_spell = torch.tensor(b.policy.logps.spell_logp, device=self.device)
                adv = torch.tensor(b.adv, device=self.device)
                returns = torch.tensor(b.returns, device=self.device)
                ratio = (prob_act / old_prob_act) * (prob_target / old_prob_target) * (prob_position / old_prob_position)

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
                e_loss = dist_act.entropy().mean() + dist_position.entropy().mean() + dist_targets.entropy().mean() # dist_spell.entropy().mean()
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

    def to_dev(self, dev):
        self.to(dev)
        self.device = dev
        self.net.device = dev
        self.net.inpipe.device = dev


class H3Agent_rewarded(H3Agent):

    def start_train(self,use_expert_data=False):
        # H3SampleCollector_SIL.check_done = check_done_2
        count = 0
        file_list = ['ENV/battles/0.json', 'ENV/battles/1.json', 'ENV/battles/2.json', 'ENV/battles/3.json','ENV/battles/4.json']
        # file_list = ['ENV/battles/0.json']
        format_postion = False
        replay_buffer = ReplayBuffer(50000, ignore_obs_next=True)
        max_sar_manager = H3ReplayManager_SIL(file_list, self, use_expert_data=use_expert_data,
                                              format_postion=format_postion)
        collector = H3SampleCollector_SIL(file_list, self, replay_buffer, max_sar_manager,
                                          format_postion=format_postion)
        cache_idx = range(len(file_list))
        if Linux:
            sample_num = 200
        else:
            sample_num = 20
        print_len = 65
        logger.info(f"start training sample eps/epoch = {sample_num}")
        while True:
            self.eval()
            self.in_train = True
            collector.buffer.reset()
            collector.reset_counts()
            for ii in range(sample_num):
                collector.collect_eps(-1, from_start=False)
            batch_data = collector.buffer.sample(0)[0]
            logger.info(len(batch_data.rew))
            self.train()
            self.in_train = True
            self.process_max_tree(batch_data)

            me_amount = np.zeros((print_len, 2))
            enemy_amount = np.zeros((print_len, 3))
            for i in range(print_len):
                stacks = batch_data.obs.attri_stack[i]
                mask = np.logical_and(stacks[:, 0] == 0, stacks[:, 3] > 1)
                me_am = stacks[mask][:, 3]
                me_amount[i, :min(2, len(me_am))] = me_am[:2]

                mask = stacks[:, 0] == 1
                enemy_am = stacks[mask][:, 3]
                enemy_amount[i, :min(3, len(enemy_am))] = enemy_am[:3]

            me_amount = me_amount.transpose().astype(np.float) / 10
            enemy_amount = enemy_amount.transpose().astype(np.float) / 10
            logger.info(batch_data.done.astype(np.float)[:print_len] * 1.1)
            logger.info("amount")
            logger.info(-batch_data.obs.attri_stack[:print_len, 0, 1].astype(np.float))
            logger.info(me_amount[0])
            logger.info(me_amount[1])
            logger.info("enemy_amount")
            logger.info(enemy_amount[0])
            logger.info(enemy_amount[1])
            logger.info(enemy_amount[2])
            logger.info("act_logp")
            logger.info(batch_data.policy.logps.act_logp[:print_len])
            logger.info(batch_data.act.act_id.astype(np.float)[:print_len])
            logger.info("position_logp")
            logger.info(batch_data.policy.logps.position_logp[:print_len])
            logger.info(-(batch_data.act.position_id.astype(np.float)[:print_len] // 17) / 10)
            logger.info(-(batch_data.act.position_id.astype(np.float)[:print_len] % 17) / 10)
            logger.info("target")
            logger.info(batch_data.act.target_id.astype(np.float)[:print_len])
            logger.info("adv")
            logger.info(batch_data.rew.astype(np.float)[:print_len])
            logger.info(batch_data.adv[:print_len])
            if Linux:
                self.to_dev("cuda")
            loss = self.learn(batch_data, batch_size=10240)
            sil_sars = max_sar_manager.get_sars()
            if len(sil_sars):
                sil_sars = Batch.cat(sil_sars)
                # self.process_gae(sil_sars, single_batch=False,sil=True)
                logger.info("sil act_logp before")
                logger.info(sil_sars.policy.logps.act_logp[:print_len])
                logger.info(sil_sars.act.act_id.astype(np.float)[:print_len])
                logger.info("sil adv before")
                logger.info(sil_sars.rew.astype(np.float)[:print_len])
                logger.info(sil_sars.policy.value[:print_len])
                loss = self.learn(sil_sars, batch_size=10240)
                self.process_gae(sil_sars, single_batch=False,sil=True)
                logger.info("sil act_logp after")
                logger.info(sil_sars.policy.logps.act_logp[:print_len])
                logger.info(sil_sars.act.act_id.astype(np.float)[:print_len])
                logger.info("sil adv after")
                logger.info(sil_sars.rew.astype(np.float)[:print_len])
                logger.info(sil_sars.policy.value[:print_len])

            self.eval()
            self.in_train = False
            if Linux:
                self.to_dev("cpu")
            '''test how many times agent can win'''
            wr, wc = collector.get_win_rates()
            logger.info(f'win rate = {wr}')
            logger.info(f'cont = {wc}')
            test_battle_idx = cache_idx
            win_count = []
            collector.reset_counts()
            for file_idx in test_battle_idx:
                collector.buffer.reset()
                ct = collector.collect_eps(file_idx,from_start=True)
                win_count.append(ct)
                logger.info(f"test-{count}-{file_list[file_idx]} win count = {ct}")
            if sum(win_count) > collector.max_win_count:
                logger.info("model saved")
                torch.save(self.state_dict(), "model_param.pkl")
                collector.max_win_count = sum(win_count)
            count += 1
            logger.info(f'count={count}')

class H3Agent_rewarded_no_sil(H3Agent):

    def start_train(self,use_expert_data=False):
        '''
        interfaces check
        '''
        # H3SampleCollector.check_done = check_done_2

        # agent.load_state_dict(torch.load("model_param.pkl"))
        count = 0
        # file_list = ['ENV/battles/0.json', 'ENV/battles/1.json', 'ENV/battles/2.json', 'ENV/battles/3.json','ENV/battles/4.json']
        file_list = ['ENV/battles/0.json']
        format_postion = False
        replay_buffer = ReplayBuffer(50000, ignore_obs_next=True)
        max_sar_manager = H3ReplayManager_SIL(file_list, self, use_expert_data=use_expert_data,format_postion=format_postion)
        collector = H3SampleCollector_SIL(file_list, self, replay_buffer, max_sar_manager,format_postion=format_postion)
        cache_idx = range(len(file_list))
        if Linux:
            sample_num = 200
        else:
            sample_num = 20
        print_len = 65
        logger.info(f"start training sample eps/epoch = {sample_num}")
        while True:
            self.eval()
            self.in_train = True
            collector.buffer.reset()
            collector.reset_counts()
            for ii in range(sample_num):
                collector.collect_eps(-1,from_start=False)
            batch_data = collector.buffer.sample(0)[0]
            sil_sars = max_sar_manager.get_sars()
            if len(sil_sars):
                sil_sars = Batch.cat(sil_sars)
            batch_data = Batch.cat([batch_data,sil_sars])
            logger.info(len(batch_data.rew))
            self.train()
            self.in_train = True
            self.process_gae(batch_data, single_batch=False)

            me_amount = np.zeros((print_len, 2))
            enemy_amount = np.zeros((print_len, 3))
            for i in range(print_len):
                stacks = batch_data.obs.attri_stack[i]
                mask = np.logical_and(stacks[:, 0] == 0,stacks[:,3] > 1)
                me_am = stacks[mask][:, 3]
                me_amount[i,:min(2,len(me_am))] = me_am[:2]

                mask = stacks[:, 0] == 1
                enemy_am = stacks[mask][:, 3]
                enemy_amount[i,:min(3,len(enemy_am))] = enemy_am[:3]

            me_amount = me_amount.transpose().astype(np.float) / 10
            enemy_amount = enemy_amount.transpose().astype(np.float) / 10
            logger.info(batch_data.done.astype(np.float)[:print_len] * 1.1)
            logger.info("amount")
            logger.info(-batch_data.obs.attri_stack[:print_len, 0, 1].astype(np.float))
            logger.info(me_amount[0])
            logger.info(me_amount[1])
            logger.info("enemy_amount")
            logger.info(enemy_amount[0])
            logger.info(enemy_amount[1])
            logger.info(enemy_amount[2])
            logger.info("act_logp")
            logger.info(batch_data.policy.logps.act_logp[:print_len])
            logger.info(batch_data.act.act_id.astype(np.float)[:print_len])
            logger.info("position_logp")
            logger.info(batch_data.policy.logps.position_logp[:print_len])
            logger.info(-(batch_data.act.position_id.astype(np.float)[:print_len] // 17) / 10)
            logger.info(-(batch_data.act.position_id.astype(np.float)[:print_len] % 17) / 10)
            logger.info("target")
            logger.info(batch_data.act.target_id.astype(np.float)[:print_len])
            logger.info("adv")
            logger.info(batch_data.rew.astype(np.float)[:print_len])
            logger.info(batch_data.adv[:print_len])
            if Linux:
                self.to_dev("cuda")
            loss = self.learn(batch_data, batch_size=2000)
            sil_sars = max_sar_manager.get_sars()
            if len(sil_sars):
                sil_sars = Batch.cat(sil_sars)
                logger.info("sil act_logp")
                logger.info(sil_sars.policy.logps.act_logp[:print_len])
                logger.info(sil_sars.act.act_id.astype(np.float)[:print_len])
                logger.info("sil adv")
                logger.info(sil_sars.rew.astype(np.float)[:print_len])
                logger.info(sil_sars.policy.value[:print_len])
                # loss = self.learn(sil_sars, batch_size=2000)
            if Linux:
                self.to_dev("cpu")
            self.eval()
            self.in_train = False
            '''test how many times agent can win'''
            wr, wc = collector.get_win_rates()
            logger.info(f'win rate = {wr}')
            logger.info(f'cont = {wc}')
            # if max_sar_manager.il_done:
            #     test_battle_idx = cache_idx
            # else:
            #     test_battle_idx = [0]
            test_battle_idx = cache_idx
            win_count = []
            collector.reset_counts()
            for file_idx in test_battle_idx:
                collector.buffer.reset()
                ct = collector.collect_eps(file_idx)
                win_count.append(ct)
                logger.info(f"test-{count}-{file_list[file_idx]} win count = {ct}")
            if sum(win_count) > max_win_count:
                logger.info("model saved")
                torch.save(self.state_dict(), "model_param.pkl")
                max_win_count = sum(win_count)
            count += 1
            logger.info(f'count={count}')

reward_def = 1.
r_cum = 0


def record_sar_ppo(self, battle, acting_stack, sars):
    if acting_stack.side == 0:
        self.acted = True
        obs = sars["obs"]
        acts = sars["act"]
        done = sars["done"]
        mask = sars["mask"]
        logps = sars["logps"]
        value = sars["value"]
        if done:
            if acting_stack.side == battle.get_winner():
                self.buffer.add(Batch(obs=obs, act=acts, rew=reward_def, done=1, info=mask,
                                      policy={"value": value, "logps": logps}))
            else:
                self.buffer.add(Batch(obs=obs, act=acts, rew=-reward_def, done=1, info=mask,
                                      policy={"value": value, "logps": logps}))
        else:
            self.buffer.add(Batch(obs=obs, act=acts, rew=0, done=0, info=mask, policy={"value": value, "logps": logps}))
    else:
        done = sars["done"]
        if done:
            if self.acted:
                last_idx = self.buffer.last_index
                if acting_stack.side == battle.get_winner():
                    self.buffer.rew[last_idx] = reward_def
                    self.buffer.done[last_idx] = 1
                else:
                    self.buffer.rew[last_idx] = -reward_def
                    self.buffer.done[last_idx] = 1
            else:
                logger.info("你的军队还没出手就被干掉了 (╯°Д°)╯︵┻━┻")

def record_sar_max_tree(collector, battle, acting_stack, sars):
    if acting_stack.side == 0:
        obs = sars["obs"]
        acts = sars["act"]
        done = sars["done"]
        mask = sars["mask"]
        attri_stack_orig = obs['attri_stack']
        obs['attri_stack'] = get_tuple(attri_stack_orig)
        if done:
            win = sars["win"]
            if win:
                reward = compute_reward(battle)
            else:
                reward = -reward_def
            # reward = compute_reward(battle)
            collector.buffer.add(Batch(obs=obs, act=acts, rew=reward, done=1, info=mask,
                                  policy={'attri_stack_orig': attri_stack_orig,'done_':1}))
        else:
            collector.buffer.add(
                Batch(obs=obs, act=acts, rew=0, done=0, info=mask,
                      policy={'attri_stack_orig': attri_stack_orig,'done_':0}))
    else:
        done = sars["done"]
        if done:
            if len(collector.buffer):
                win = sars["win"]
                if win:
                    reward = compute_reward(battle)
                else:
                    reward = -reward_def
                # reward = compute_reward(battle)
                last_idx = collector.buffer.last_index[0]
                collector.buffer.rew[last_idx] = reward
                collector.buffer.done[last_idx] = 1
                collector.buffer.policy.done_[last_idx] = 1
            else:
                logger.info("你的军队还没出手就被干掉了 (╯°Д°)╯︵┻━┻")

def compute_reward(battle: Battle):
    att_HP = [st.amount_base * st.health for st in battle.attacker_stacks]
    def_HP = [st.amount_base * st.health for st in battle.defender_stacks]
    att_HP_left = [(st.amount - 1) * st.health + st.first_HP_Left for st in battle.attacker_stacks if
                   st.amount > 0]
    def_HP_left = [(st.amount - 1) * st.health + st.first_HP_Left for st in battle.defender_stacks if
                   st.amount > 0]
    reward = reward_def * (sum(att_HP_left) / sum(att_HP) - sum(def_HP_left) / sum(def_HP))
    return reward
class H3SampleCollector:
    def __init__(self,record_sar_func,file_list,agent:H3Agent,full_buffer:ReplayBuffer,format_postion=False,win_count_size = 10):
        self.record_sar:record_sar_max_tree = record_sar_func
        self.agent = agent
        self.buffer = full_buffer
        cfg = EasyDict(replay_buffer_size=100, deepcopy=False, exp_name='test_episode_buffer',periodic_thruput_seconds=60,
                       enable_track_used_data=False)
        self.current_buffer = []
        cfg.replay_buffer_size = 1000
        self.history_buffer = EpisodeReplayBuffer(
            cfg, exp_name=cfg.exp_name, instance_name='history_buffer'
        )
        self.max_win_count = 0
        self.best_hist_buffer = []
        self.acted = False
        fl = len(file_list)
        self.defender_states = [] #type:List[BStack]
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
        self.win_count_size = win_count_size
        self.win_count = np.zeros((fl,win_count_size),dtype=int)
        self.count = np.ones((fl,))
    def collect_1_ep(self,file=None,battle:Battle=None,n_step = 200,print_act = False,td=False):
        if not battle:
            battle = Battle(agent=self.agent)
            battle.load_battle(file)
        battle.checkNewRound()
        had_acted = False
        win = False
        for ii in range(n_step):
            acting_stack = battle.cur_stack
            if acting_stack.side == 0:
                print_act = print_act and ii < 5
                battle_action, sars = acting_stack.active_stack(ret_obs=True, print_act=print_act)
            else:
                battle_action = acting_stack.active_stack()
                sars = {}  # sars = {"act":None,"obs":None,"mask":None,"logps":None,"value":0}
            if acting_stack.side == 0:
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
            done,win = self.check_done(battle)
            #buffer sar
            if had_acted:
                sars["done"] = done
                sars["win"] = win
                self.record_sar(self,battle,acting_stack,sars)
            #get winner
            if done:
                break
        return win

    '''[single side Wheel fight]'''
    def collect_eps(self,file_idx,battle:Battle=None,n_step = 200,print_act = False,td=False,from_start=False):
        arena = Battle(by_AI=[2, 1], agent=self.agent)
        file_idx,obs_idx, obs, defender_stacks = self.choose_start(file_idx,from_start=from_start)
        '''record start point'''
        # buf_start = self.buffer._index
        if obs_idx > 0:
            self.prepare_obs(file_idx,obs_idx)
        arena.load_battle(obs, load_ai_side=False, format_postion=False)
        win_index = -1
        episode = []
        for r in range(10):
            self.buffer.reset()
            win = self.collect_1_ep(battle=arena)
            r'''首场不论胜负,接下来只记录胜场'''
            if win:
                traj,indice = self.buffer.sample(0)
                episode.append(traj)
                if r == 0 and obs_idx > 0:
                    '''normal start after middle start need refresh inner states of stacks'''
                    for st in defender_stacks:
                        st.in_battle = arena
                    arena.defender_stacks = defender_stacks
                arena.split_army()
            else:
                if r == 0:
                    traj, indice = self.buffer.sample(0)
                    episode.append(traj)
                break
        # if win_index > 0:
        #     self.buffer.done[win_index] = 1
        # self.buffer.done[self.buffer.last_index[0]] = 1
        # episode,indice = self.buffer.sample(0)
        win_count = len(episode)
        if win_count > 1:
            for ei in range(win_count - 1):
                episode[ei].done[-1] = 0
                episode[ei].rew[-1] = reward_def
                episode[ei].obs_next[-1] = episode[ei+1].obs[0]
        episode = Batch.cat(episode)
        self.current_buffer.append(episode)
        # win_count = self.update_count(file_idx,obs_idx,self.buffer,0,self.buffer._index)
        # if win_count > 0:
        #     cumulate_reward(self.buffer,buf_start,self.buffer._index)
        # if from_start and self.max_win_count - win_count > 3:
        #     max_data = self.buffer[buf_start:self.buffer._index]
        #     dump_episo([max_data.obs, max_data.act, max_data.rew, max_data.done, max_data.info], "ENV/max_sars",
        #                f'max_data_{file_idx}.npy')
        #     logger.info(f"diversed states dumped in max_data_{file_idx}.npy")
        #     torch.save(self.agent.state_dict(), "model_param.pkl")
        #     logger.info("model saved")
        #     assert False
        return win_count
    def prepare_obs(self,file_idx,obs_idx):
        raise Exception('prepare_obs should be implemented')
    def choose_start(self, idx, from_start=False):
        return idx,-1, self.file_list_cache[idx], None  # RL
    def check_done(self,battle:Battle):
        '''single troop get killed over 40% or shooter killed over 20%'''
        over_killed = False
        done = False
        win = False
        for st in battle.merge_stacks(copy_stack=True):
            if st.is_shooter and st.amount < st.amount_base * 0.8:
                done = True
                over_killed = True
                break
            elif st.amount < st.amount_base * 0.6:
                done = True
                over_killed = True
                break
        if not over_killed:
            done = battle.check_battle_end()
            if done:
                win = True
        return done,win
    def reset_count(self,idx):
        self.win_count[idx] *= 0
        self.count[idx] = 1
    def reset_counts(self):
        for i in range(len(self.count)):
            self.reset_count(i)
    def get_win_rate(self,idx):
        return self.win_count[idx] / self.count[idx],self.count[idx]
    def get_win_rates(self):
        return self.win_count / np.expand_dims(self.count,axis=1) , self.count

    def update_count(self,idx,obs_idx,data:ReplayBuffer,start,end):
        self.count[idx] += 1
        if end == start:
            return 0 # lose without a act
        elif end < start:
            logger.info('buff size is not enough~')
            return min(int(sum(data.done[start:])),self.win_count_size)
        else:
            assert data.done[end-1] > 0
            data_win_sum = int(sum(data.done[start:end]))
            if data.rew[end - 1] < 0:
                data_win_sum -= 1
            data_win_sum = min(data_win_sum,self.win_count_size)
            if data_win_sum >  0:
                self.win_count[idx,data_win_sum - 1] += 1
            return data_win_sum

class H3SampleCollector_SIL(H3SampleCollector):
    def __init__(self,file_list,agent:H3Agent,full_buffer:ReplayBuffer,max_sar_manager:H3ReplayManager_SIL,format_postion=False):
        super(H3SampleCollector_SIL, self).__init__(file_list,agent,full_buffer,format_postion)
        self.max_sar_manager = max_sar_manager
        self.start_HP = None
        self.max_win_count = 0 if not max_sar_manager.max_sar[0] else sum(max_sar_manager.max_sar[0].done)

    def choose_start(self, idx = -1, from_start=False):
        return self.max_sar_manager.choose_start(idx,from_start=from_start)

    def update_count(self, idx, obs_idx, data: ReplayBuffer, start, end):
        win_count = super(H3SampleCollector_SIL, self).update_count(idx, obs_idx, data, start, end)
        self.max_sar_manager.update_max(idx, obs_idx, data, start, end)
        return win_count

    def prepare_obs(self,file_idx,obs_idx):
        self.start_HP = self.get_start_HP(self.max_sar_manager.max_sar[file_idx][0])
        traj = self.max_sar_manager.max_sar[file_idx][:obs_idx]
        self.agent.process_gae(traj, single_batch=False, sil=True)
        for i in range(obs_idx):
            step = traj[i]
            self.buffer.add(Batch(obs=step.obs, act=step.act, rew=step.rew, done=step.done, info=step.info,policy={"value": step.policy.value, "logps": step.policy.logps}))
    def get_start_HP(self,sar):
        ep = sar.obs.attri_stack
        start_mask = np.logical_and(ep[:, 0] == 0,ep[:, 3] > 0)
        start_stack = ep[start_mask]
        start_HP_att = ((start_stack[:, 3] - 1) * start_stack[:, 6] + start_stack[:, 5]).sum()
        start_mask = ep[:, 0] == 1
        start_stack = ep[start_mask]
        start_HP_def = ((start_stack[:, 3] - 1) * start_stack[:, 6] + start_stack[:, 5]).sum()
        return (start_HP_att,start_HP_def)
    def compute_reward(self,battle: Battle):
        if self.start_HP:
            att_HP = [self.start_HP[0]]
            def_HP = [self.start_HP[1]]
            self.start_HP = None
        else:
            att_HP = [st.amount_base * st.health for st in battle.attacker_stacks]
            def_HP = [st.amount_base * st.health for st in battle.defender_stacks]
        att_HP_left = [(st.amount - 1) * st.health + st.first_HP_Left for st in battle.attacker_stacks if
                       st.amount > 0]
        def_HP_left = [(st.amount - 1) * st.health + st.first_HP_Left for st in battle.defender_stacks if
                       st.amount > 0]
        reward = reward_def * (sum(att_HP_left) / sum(att_HP) - sum(def_HP_left) / sum(def_HP))
        return reward
    # def record_sar(self, operator, battle: Battle, acting_stack, sars):
    #     if acting_stack.side == 0:
    #         self.acted = True
    #         obs = sars["obs"]
    #         acts = sars["act"]
    #         done = sars["done"]
    #         mask = sars["mask"]
    #         logps = sars["logps"]
    #         value = sars["value"]
    #         attri_stack_orig = obs['attri_stack']
    #         obs['attri_stack'] = get_tuple(attri_stack_orig)
    #         if done:
    #             win = sars["win"]
    #             if win:
    #                 reward = self.compute_reward(battle)
    #             else:
    #                 reward = -reward_def
    #             self.buffer.add(Batch(obs=obs, act=acts, rew=reward, done=1, info=mask,policy={"value": value, "logps": logps,'attri_stack_orig':attri_stack_orig}))
    #         else:
    #             self.buffer.add(
    #                 Batch(obs=obs, act=acts, rew=0, done=0, info=mask, policy={"value": value, "logps": logps,'attri_stack_orig':attri_stack_orig}))
    #     else:
    #         done = sars["done"]
    #         if done:
    #             if self.acted:
    #                 win = sars["win"]
    #                 if win:
    #                     reward = self.compute_reward(battle)
    #                 else:
    #                     reward = -reward_def
    #                 last_idx = self.buffer.last_index
    #                 self.buffer.rew[last_idx] = reward
    #                 self.buffer.done[last_idx] = 1
    #             else:
    #                 logger.info("你的军队还没出手就被干掉了 (╯°Д°)╯︵┻━┻")

class H3SampleCollector_expert(H3SampleCollector):
    def __init__(self,*param1,max_sar_manager:H3ReplayManager_SIL=None,**param2):
        super(H3SampleCollector_expert, self).__init__(*param1,**param2)
        # self.max_sar_manager = max_sar_manager

    def collect_1_ep(self, file=None, battle: Battle = None, n_step=200, print_act=False, td=False):
        from ENV import H3_battleInterface
        import pygame
        # 初始化游戏
        pygame.init()  # 初始化pygame
        pygame.display.set_caption('This is my first pyVCMI')  # 设置窗口标题
        if not battle:
            battle = Battle(by_AI=[0, 1])
            battle.load_battle("ENV/battles/0.json", shuffle_postion=False, load_ai_side=False)
        battle.checkNewRound()
        bi = H3_battleInterface.BattleInterface(battle)
        logps, value = None, 0
        had_acted = False
        acting_stack = battle.cur_stack
        # FIXME fake act at first
        bi.next_act = acting_stack.active_stack()
        # 事件循环(main loop)
        while bi.running:
            do_act = bi.handleEvents()
            if do_act:
                if acting_stack.side == 0:
                    if not had_acted:
                        # for st in battle.stacks:
                        #     battle.ai_value[st.side] += st.ai_value * st.amount
                        had_acted = True
                    obs, acts, mask = self.get_act_info(battle, bi.next_act)
                else:
                    obs, acts, mask = None, None, None
                damage_dealt, damage_get, killed_dealt, killed_get = battle.doAction(bi.next_act)
                # check done
                done, win = self.check_done(battle)
                # buffer sar
                if had_acted:
                    sars = {"act":acts,"obs":obs,"mask":mask,"done": done}
                    sars["done"] = done
                    sars["win"] = win
                    self.record_sar(self, battle, acting_stack, sars)
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
    def collect_eps(self,file_idx,battle:Battle=None,n_step = 200,print_act = False,td=False):
        arena = Battle(by_AI=[0, 1])
        fidx,obs_idx, obs, defender_stacks = self.choose_start(file_idx)
        arena.load_battle(obs, load_ai_side=False, format_postion=False)
        arena.checkNewRound()
        self.collect_1_ep(battle=arena)
        #
        # arena.split_army()
        # arena.checkNewRound()
        # self.collect_1_ep(battle=arena)
        #
        # arena.split_army()
        # arena.checkNewRound()
        self.collect_1_ep(battle=arena)
        batch_rew = self.buffer.rew[:self.buffer._index]
        data = self.buffer.sample(0)[0]
        dump_episo([data.obs, data.act, data.rew, data.done, data.info], "ENV/episode", file='0.npy')
    @staticmethod
    def get_act_info(battle, act):
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
                assert st.is_alive(), f"{st} is dead?"
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
        return obs, acts, mask
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
    # print(grad.abs().sum(dim=-1))
    print(grad)

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
                if winner == 0:
                    test_win += 1
                break
    return test_win/iter_N


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
def load_episo(dir,file=None):
    if file:
        npys = [file]
    else:
        npys = [f for f in os.listdir(dir) if f.endswith(".npy")]
    if len(npys) == 0:
        return
    expert = []
    for f in npys:
        tmp_f = os.path.join(dir,f)
        obs_orig,act, rew, done, mask = np.load(tmp_f, allow_pickle=True)
        obs = Batch.stack(obs_orig)
        act = Batch.stack(act)
        mask = Batch.stack(mask)
        obs_next = Batch(obs,copy=True)
        obs_next[:-1] = obs[1:]
        episode = Batch(obs=obs, obs_next=obs_next, act=act, rew=rew, done=done, info=mask)
        expert.append(episode)
    return expert
def start_test(file):
    import pygame
    pygame.init()  # 初始化pygame
    pygame.display.set_caption('This is my first pyVCMI')  # 设置窗口标题

    model = H3_Q_net(dev)
    # model.act_ids.weight.register_hook(hook_me)
    agent = H3AgentQ(model)
    agent.load_state_dict(torch.load("model_param.pkl"))
    agent.eval()
    agent.model = 1
    from ENV.H3_battleInterface import start_game_s_gui
    battle = Battle(agent=agent)
    battle.load_battle(file=file)
    bi = start_game_s_gui(battle=battle)
    battle = Battle(agent=agent)
    battle.load_battle(file=file)
    start_game_s_gui(battle=battle,battle_int=bi)

def check_done_2(self,battle):
    done = battle.check_battle_end()
    win = None
    if done:
        win = battle.get_winner() == 0
    return done,win
def update_count_2(self,idx,obs_idx,data:ReplayBuffer,start,end):
    assert data.done[end-1] > 0
    # assert data.rew[end-1] > 0
    if end > start:
        data_win_sum = int(sum(data.done[start:end]))
    else:
        indice = list(range(start, len(data))) + list(range(end))
        data_win_sum = int(sum(data.done[indice]))
    data_win_sum = min(data_win_sum,self.win_count_size)
    if data_win_sum > 0:
        self.win_count[idx,data_win_sum - 1] += 1
    return data_win_sum

def update_max_2(self,idx,obs_idx,data:ReplayBuffer,start,end,obs_bias):
    if end > start:
        data_rew = data.rew[start:end]
        indice = range(start,end)
    else:
        indice = list(range(start, len(data))) + list(range(end))
        data_rew = data.rew[idx]
    data_rew = sum(data_rew)
    if not self.max_sar[idx]:
        if data_rew > 0:
            self.max_sar[idx] = data[indice]
            max_data = self.max_sar[idx]
            dump_episo([max_data.obs, max_data.act, max_data.rew, max_data.done, max_data.info], "ENV/max_sars",
                       f'max_data_{idx}.npy')
            logger.info(f"states dumped in max_data_{idx}.npy")
    else:
        record_done = int(sum(self.max_sar[idx].done))
        data_done = int(sum(data.done[indice]))
        record_rew = sum(self.max_sar[idx].rew)
        if record_rew < data_rew or record_done < data_done:
            if obs_idx <= 0:
                logger.info(f"idx-{idx} get new traj from scratch")
            else:
                logger.info(f"idx-{idx} get new traj")
            self.max_sar[idx] = data[indice]
            max_data = self.max_sar[idx]
            dump_episo([max_data.obs, max_data.act, max_data.rew, max_data.done, max_data.info], "ENV/max_sars",f'max_data_{idx}.npy')
            logger.info(f"states dumped in max_data_{idx}.npy")
def cumulate_reward(buffer,start,end):
    assert end != start
    assert buffer.done[end - 1] > 0
    tmp = None
    if buffer.rew[end - 1] < 0:
        tmp = buffer.rew[end - 1]
        buffer.rew[end - 1] = 0
    if end > start:
        batch_rew = buffer.rew[start:end]
        batch_done = buffer.done[start:end]
        acc_rewards = np.add.accumulate(batch_rew[batch_done > 0][::-1])
        batch_rew[batch_done > 0] = acc_rewards[::-1]
    else:
        idx = list(range(start, len(buffer))) + list(range(end))
        batch_rew = buffer.rew[idx]
        batch_done = buffer.done[idx]
        acc_rewards = np.add.accumulate(batch_rew[batch_done > 0][::-1])
        batch_rew[batch_done > 0] = acc_rewards[::-1]
        buffer.rew[idx] = batch_rew
    if tmp:
        buffer.rew[end - 1] = tmp
#TODO done state needs next state....
def compute_reward_from_episodes(buffer, start, end):
    if end == start:
        logger.info(f"end == start == {end}???")
        return
    if end > start:
        batch_rew = buffer.rew[start:end]
        batch_done = buffer.done[start:end]
        batch_obs = buffer.obs.attri_stack[start:end]
        done_index = np.argwhere(batch_done> 0).squeeze()
    else:
        idx = list(range(start, len(buffer))) + list(range(end))
        batch_rew = buffer.rew[idx]
        batch_done = buffer.done[idx]
        done_index = np.argwhere(batch_done> 0).squeeze()
    if len(done_index) == 1:
        start_index = np.array([0])
        end_index = done_index
    else:
        start_index = np.append(0,(done_index+ 1)[:-1])
        end_index = np.append((done_index+ 1)[:-1],done_index[-1])
    start_obs = batch_obs[start_index]
    end_obs = batch_obs[end_index]
    start_HP_att = np.zeros((len(done_index),))
    start_HP_def = np.zeros((len(done_index),))
    end_HP_att = np.zeros((len(done_index),))
    end_HP_def = np.zeros((len(done_index),))
    for i,ep in enumerate(start_obs):
        start_mask = ep[:,0] == 0
        start_stack = ep[start_mask]
        start_HP_att[i] = ((start_stack[:,3] - 1) * start_stack[:,6] + start_stack[:,5]).sum()
        start_mask = ep[:, 0] == 1
        start_stack = ep[start_mask]
        start_HP_def[i] = ((start_stack[:,3] - 1) * start_stack[:,6] + start_stack[:,5]).sum()
    for i,ep in enumerate(end_obs):
        end_mask = ep[:,0] == 0
        end_stack = ep[end_mask]
        end_HP_def[i] = 0.
        end_HP_att[i] = ((end_stack[:,3] - 1) * end_stack[:,6] + end_stack[:,5]).sum()

    HP_reward = reward_def * (end_HP_att / start_HP_att - end_HP_def / start_HP_def)
    # HP_reward = np.add.accumulate(HP_reward[::-1])
    batch_rew[done_index] = HP_reward
    if end < start:
        idx = list(range(start, len(buffer))) + list(range(end))
        buffer.rew[idx] = batch_rew
#@profile
def start_train(use_expert_data=False):
    # 初始化 agent
    model = H3_Q_net(dev)
    # model.act_ids.weight.register_hook(hook_me)
    agent = H3AgentQ(model)
    agent.start_train() #use_expert_data=use_expert_data



def start_game_record_s():
    replay_buffer = ReplayBuffer(10000, ignore_obs_next=True)
    file_list = ['ENV/battles/0.json']
    collector = H3SampleCollector_expert(record_sar_max_tree,file_list, None, replay_buffer,format_postion=False)
    collector.collect_eps(0)
def start_replay_m(dir,file=None):
    from ENV.H3_battleInterface import start_replay
    data = load_episo(dir,file)
    start_replay(data[0])
M=0
if __name__ == '__main__':
    if Linux:
        start_train(use_expert_data=True)
    else:
        # start_game_record_s()
        # start_replay_m("ENV/max_sars",'max_data_0.npy')  #"ENV/max_sars" episode
        start_train(use_expert_data=False)
        # start_test(file="ENV/battles/0.json")

