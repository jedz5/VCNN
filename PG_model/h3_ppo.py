
import sys
sys.path.extend(['/home/enigma/work/project/VCNN', 'D:/project/VCNN/'])
from PG_model.discrete_net import H3_net
from torch import nn
from typing import Tuple, Optional
from ENV.H3_battle import *
from tianshou.policy import PGPolicy
from tianshou.data import Batch, ReplayBuffer

# import pdb

np.set_printoptions(precision=2,suppress=True)
logger = get_logger()[1]
dev = 'cpu'
def softmax(logits, mask_orig,dev,add_little = False,in_train = True):
    mask = torch.tensor(mask_orig,dtype=torch.float,device=dev)
    if in_train:
        logits1 = logits.sub(logits.max(dim=-1,keepdim=True)[0]).exp()
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
                 optim: torch.optim.Optimizer,
                 dist_fn: torch.distributions.Distribution,
                 device,
                 discount_factor: float = 0.9,
                 max_grad_norm: Optional[float] = 0.5,
                 eps_clip: float = .2,
                 vf_coef: float = .5,
                 ent_coef: float = .01,
                 action_range: Optional[Tuple[float, float]] = None,
                 gae_lambda: float = 0.92,
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
            for b in batch.split(self._batch_size, shuffle=False):
                v_.append(self.ppo_net(**b.obs_next,critic_only = True))
        v_ = torch.cat(v_, dim=0).cpu().numpy()
        return self.compute_episodic_return(
            batch, v_, gamma=self._gamma, gae_lambda=self._lambda)

    #单次输入 obs->act,mask
    def forward(self, ind,attri_stack,planes_stack,plane_glb,
                env,shooter = False,print_act = False,
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
        if self.in_train:
            act_prob = softmax(act_logits, mask_acts, self.device)
            if print_act:
                logger.info(act_prob)
                logger.info(value)
            if torch.sum(act_prob, dim=-1, keepdim=True).item() > 0.9:
                act_id = self.dist_fn(act_prob).sample()[0].item()
            else:
                act_prob = softmax(act_logits, mask_acts, self.device, in_train=False)
                act_id = torch.argmax(act_prob, dim=-1)[0].item()
            if act_id == action_type.move.value:
                mask_position = env.legal_act(level=1, act_id=act_id)
                position_prob = softmax(position_logits, mask_position, self.device)
                if torch.sum(position_prob,dim=-1,keepdim=True).item() > 0.9:
                    position_id = self.dist_fn(position_prob).sample()[0].item()
                else:
                    position_prob = softmax(position_logits, mask_position, self.device, in_train=False)
                    position_id = torch.argmax(position_prob, dim=-1)[0].item()
            elif act_id == action_type.attack.value:
                mask_targets = env.legal_act(level=1, act_id=act_id)
                targets_prob = softmax(targets_logits, mask_targets, self.device)
                if torch.sum(targets_prob,dim=-1,keepdim=True).item() > 0.9:
                    target_id = self.dist_fn(targets_prob).sample()[0].item()
                else:
                    targets_prob = softmax(targets_logits, mask_targets, self.device, in_train=False)
                    target_id = torch.argmax(targets_prob, dim=-1)[0].item()
                if not shooter:
                    mask_position = env.legal_act(level=2, act_id=act_id, target_id=target_id)
                    position_prob = softmax(position_logits, mask_position, self.device)
                    if torch.sum(position_prob,dim=-1,keepdim=True).item() > 0.9:
                        position_id = self.dist_fn(position_prob).sample()[0].item()
                    else:
                        position_prob = softmax(position_logits, mask_position, self.device,in_train=False)
                        position_id = torch.argmax(position_prob, dim=-1)[0].item()
        else:
            act_logits = softmax(act_logits, mask_acts, self.device,in_train=False)
            if print_act:
                logger.info(act_logits)
                logger.info(value)
            act_id = torch.argmax(act_logits,dim=-1)[0].item()
            if act_id == action_type.move.value:
                mask_position = env.legal_act(level=1, act_id=act_id)
                position_logits = softmax(position_logits, mask_position, self.device,in_train=False)
                position_id = torch.argmax(position_logits,dim=-1)[0].item()
            elif act_id == action_type.attack.value:
                mask_targets = env.legal_act(level=1, act_id=act_id)
                targets_prob = softmax(targets_logits, mask_targets, self.device,in_train=False)
                target_id = torch.argmax(targets_prob, dim=-1)[0].item()
                mask_position = env.legal_act(level=2, act_id=act_id, target_id=target_id)
                if not shooter:
                    position_logits = softmax(position_logits, mask_position, self.device,in_train=False)
                    position_id = torch.argmax(position_logits, dim=-1)[0].item()
        return {'act_id': act_id, 'spell_id': spell_id, 'target_id': target_id, 'position_id': position_id,
                'mask_acts': mask_acts, 'mask_spell': mask_spell, 'mask_targets': mask_targets,
                'mask_position': mask_position, 'value': value}

    #TODO 5 mcts结构有何优势
    #批量输入训练
    def learn(self, batch, batch_size=None, repeat=2, **kwargs):
        # losses, clip_losses, vf_losses, ent_losses = [], [], [], []
        v = []
        old_prob_act = []
        old_prob_position = []
        old_prob_target = []
        old_prob_spell = []
        with torch.no_grad():
            for b in batch.split(self._batch_size,shuffle=False):
                # obs = {'ind': ind, 'attri_stack': attri_stack, 'planes_stack': planes_stack, 'plane_glb': plane_glb}
                # acts = {'act_id': act_id, 'position_id': position_id, 'target_id': target_id, 'spell_id': spell_id}
                # mask = {'mask_acts': result['mask_acts'], 'mask_spell': result['mask_spell'],
                #         'mask_targets': result['mask_targets'], 'mask_position': result['mask_position']}
                mask_acts = b.info.mask_acts
                mask_spell = b.info.mask_spell
                mask_targets = b.info.mask_targets
                mask_position = b.info.mask_position

                act_logits, targets_logits, position_logits, spell_logits, value = self.ppo_net(**b.obs)
                v.append(value.squeeze(-1))
                act_logits = softmax(act_logits, mask_acts, self.device,add_little=True)
                # act_logits = act_logits.softmax(dim=-1)
                old_prob_act.append(self.dist_fn(act_logits).log_prob(torch.tensor(b.act.act_id, device=self.device)))
                targets_logits = softmax(targets_logits, mask_targets, self.device,add_little=True)
                # targets_logits = targets_logits.softmax(dim=-1)
                old_prob_target.append(self.dist_fn(targets_logits).log_prob(torch.tensor(b.act.target_id, device=self.device)))
                position_logits = softmax(position_logits, mask_position, self.device,add_little=True)
                # position_logits = position_logits.softmax(dim=-1)
                old_prob_position.append(self.dist_fn(position_logits).log_prob(torch.tensor(b.act.position_id, device=self.device)))
                spell_logits = softmax(spell_logits, mask_spell, self.device,add_little=True)
                # spell_logits = spell_logits.softmax(dim=-1)
                old_prob_spell.append(self.dist_fn(spell_logits).log_prob(torch.tensor(b.act.spell_id, device=self.device)))
        try:
            batch.v = torch.cat(v, dim=0)  # old value
        except:
            print(v)
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
        pcount = 0
        for _ in range(repeat):
            for b in batch.split(batch_size,shuffle=False):
                mask_acts = b.info.mask_acts
                mask_spell = b.info.mask_spell
                mask_targets = b.info.mask_targets
                mask_position = b.info.mask_position
                act_logits, targets_logits, position_logits, spell_logits, value = self.ppo_net(**b.obs)
                act_logits = softmax(act_logits, mask_acts, self.device,add_little=True)
                # act_logits = act_logits.softmax(dim=-1)
                dist_act = self.dist_fn(act_logits)
                prob_act = dist_act.log_prob(torch.tensor(b.act.act_id, device=self.device))
                targets_logits = softmax(targets_logits, mask_targets, self.device,add_little=True)
                # targets_logits = targets_logits.softmax(dim=-1)
                dist_targets = self.dist_fn(targets_logits)
                prob_target = dist_targets.log_prob(torch.tensor(b.act.target_id, device=self.device))
                position_logits = softmax(position_logits, mask_position, self.device,add_little=True)
                # position_logits = position_logits.softmax(dim=-1)
                dist_position = self.dist_fn(position_logits)
                prob_position = dist_position.log_prob(torch.tensor(b.act.position_id, device=self.device))
                spell_logits = softmax(spell_logits, mask_spell, self.device,add_little=True)
                # spell_logits = spell_logits.softmax(dim=-1)
                dist_spell = self.dist_fn(spell_logits)
                prob_spell = dist_spell.log_prob(torch.tensor(b.act.spell_id, device=self.device))
                #TODO 如果old_prob太小？？？
                ratio = (prob_act + prob_target + prob_position + prob_spell - b.old_prob_act - b.old_prob_target - b.old_prob_position - b.old_prob_spell).exp().float()
                surr1 = ratio * b.adv
                surr2 = ratio.clamp(
                    1. - self._eps_clip, 1. + self._eps_clip) * b.adv
                if self._dual_clip:
                    clip_loss = -torch.max(torch.min(surr1, surr2),
                                           self._dual_clip * b.adv).mean()
                else:
                    clip_loss = -torch.min(surr1, surr2).mean()
                # clip_losses.append(clip_loss.item())
                if self._value_clip:
                    v_clip = b.v + (value - b.v).clamp(
                        -self._eps_clip, self._eps_clip)
                    vf1 = (b.returns - value).pow(2)
                    vf2 = (b.returns - v_clip).pow(2)
                    vf_loss = .5 * torch.max(vf1, vf2).mean()
                else:
                    vf_loss = .5 * (b.returns - value).pow(2).mean()
                # vf_losses.append(vf_loss.item())
                e_loss = (dist_act.entropy().mean() + dist_position.entropy().mean() + dist_targets.entropy().mean() + dist_spell.entropy().mean())/4
                # ent_losses.append(e_loss.item())
                #TODO 8 entropy工作原理
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
        # return {
        #     'loss': losses,
        #     'loss/clip': clip_losses,
        #     'loss/vf': vf_losses,
        #     'loss/ent': ent_losses,
        # }
wf_flag = False
def collect_eps(agent,file,buffer,n_step = 200,n_episode = 1,print_act = False):
    battle = Battle(agent=agent)
    battle.load_battle(file)
    # if random.randint(0,1):
    #     #     init_stack_position(battle)
    #     # init_stack_position(battle)
    battle.checkNewRound()
    had_acted = False
    win = False
    for ii in range(n_step):
        acting_stack = battle.cur_stack
        if acting_stack.by_AI != 2:
            battle_action = acting_stack.active_stack()
            obs, acts, mask = None,None,None
        else:
            print_act = print_act and ii < 5
            battle_action, obs, acts, mask = acting_stack.active_stack(ret_obs=True, print_act = print_act)
            had_acted = True
        battle.doAction(battle_action)
        battle.checkNewRound()
        #battle.curStack had updated
        done = battle.check_battle_end()
        reward = -0.01
        if done:
            if battle.by_AI[battle.get_winner()] == 2:
                reward = 1
                win = True
            if acting_stack.by_AI != 2:
                if had_acted:
                    buffer.rew[len(buffer) - 1] = reward
                    buffer.done[len(buffer) - 1] = True
            else:
                buffer.add(obs=obs, act=acts, rew=reward, done=True, info=mask)
            break
        if acting_stack.by_AI == 2:
            buffer.add(obs=obs, act=acts, rew=reward, done=done,info=mask)
    #TODO 0 修复
    if len(buffer) != 0:
        buffer.done[len(buffer) - 1] = True
    return win
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
def start_train():
    # 初始化 agent
    actor_critic = H3_net(dev)
    optim = torch.optim.Adam(actor_critic.parameters(), lr=0.005)
    dist = torch.distributions.Categorical
    agent = H3_policy(actor_critic,optim,dist,device=dev,gae_lambda=0.95)
    agent.load_state_dict(torch.load("model_param.pkl"))
    buffer = ReplayBuffer(10000,ignore_obs_next=True)
    count = 0
    f_max = 1
    new_start = 0
    five_done = 3
    ok = five_done
    sep = 1
    #TODO 9 SAC算法
    while True:
        agent.eval()
        agent.in_train = True
        for ii in range(500):
            bingo = np.random.choice([0,1,2],p=[0.6,0.25,0.15])
            if bingo == 0:
                agent.in_train = True
                fn = new_start
            if bingo == 1:
                agent.in_train = True
                fn = min(new_start+sep,f_max)
            if bingo == 2:
                agent.in_train = False
                fn = min(new_start+sep,f_max)
            # fn = random.randint(new_start,min(new_start+1,f_max))
            file = f'ENV/battles/{fn}.json'
            print_act = False
            if ii < 3 :
                print_act = True
                logger.info(f"------------------------{fn}.json")
            win = collect_eps(agent,file,buffer,n_episode=ii,print_act=print_act)
            if win and ii < 50:
                logger.info(f"win {fn}")
        batch_data, indice = buffer.sample(0)
        logger.info(len(buffer))
        agent.train()
        agent.in_train = True
        batch_data = agent.process_gae(batch_data)
        # v_ = []
        # with torch.no_grad():
        #     v_ = agent.ppo_net(**batch_data.obs, critic_only=True)
        # print(batch_data.done.astype(np.int))
        # print(batch_data.returns)
        # print(v_.squeeze())
        # print(batch_data.act.act_id.astype(np.int))
        if Linux:
            to_dev(agent, "cuda")
            #TODO 7 batch size?
        loss = agent.learn(batch_data,batch_size=2000)
        # with torch.no_grad():
        #     v_ = agent.ppo_net(**batch_data.obs, critic_only=True)
        # print(v_.squeeze())
        if Linux:
            to_dev(agent, "cpu")
        agent.eval()
        agent.in_train = False

        #no GUI five done
        file = f'ENV/battles/{new_start}.json'
        ct = start_game_noGUI(file, agent=agent)
        logger.info(f"test-{count}-{new_start}.json win rate = {ct}")
        if ct > 0.9:
            ok -= 1
            if ok == 0:
                new_start -= sep
                torch.save(agent.state_dict(),"model_param.pkl")
                logger.info("model saved")
                ok = five_done
                # if new_start == 0:
                #     new_start = 1
                if new_start < 0:
                    sys.exit(0)
        else:
            ok = five_done
        buffer.reset()
        if count == 2500:
            sys.exit(-1)
        count += 1
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
def start_game_noGUI(file,agent = None,by_AI = [2,1]):
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
    start_game("ENV/battles/1.json", by_AI=[2, 1],agent=agent)

def main():
    start_train()
    # start_test()
if __name__ == '__main__':
    main()