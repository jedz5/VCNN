
import sys
sys.path.extend(['/home/enigma/work/project/VCNN', 'D:/project/VCNN/'])
from PG_model.discrete_net import H3_net
from torch import nn
from typing import Tuple, Optional
from ENV.H3_battle import *
from tianshou.policy import PGPolicy
from tianshou.data import Batch, ReplayBuffer
import scipy.signal
# import pdb

np.set_printoptions(precision=2,suppress=True)
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

    def process_gae(self, batch: Batch, single_batch=True) -> Batch:
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

        if single_batch:
            rr = np.append(batch.rew, 0)
            vv = np.append(batch.policy.value, 0)
            delta = rr[:-1] + vv[1:] * gamma - vv[:-1]
            batch.adv = scipy.signal.lfilter([1], [1, -gamma * gae_lambda], delta[::-1], axis=0)[::-1]
            batch.returns = scipy.signal.lfilter([1], [1, -gamma], batch.rew[::-1], axis=0)[::-1]
        else:
            v_ = []
            old_prob_act = []
            old_prob_position = []
            old_prob_target = []
            old_prob_spell = []
            with torch.no_grad():
                for b in batch.split(self._batch_size, shuffle=False):
                    act_logits, targets_logits, position_logits, spell_logits, value = self.ppo_net(**b.obs)
                    v_.append(value.squeeze(-1))
                    act_logits = softmax(act_logits, b.info.mask_acts, self.device,add_little=True)
                    # act_logits = act_logits.softmax(dim=-1)
                    old_prob_act.append(self.dist_fn(act_logits).log_prob(torch.tensor(b.act.act_id, device=self.device)))
                    targets_logits = softmax(targets_logits, b.info.mask_targets, self.device,add_little=True)
                    # targets_logits = targets_logits.softmax(dim=-1)
                    old_prob_target.append(self.dist_fn(targets_logits).log_prob(torch.tensor(b.act.target_id, device=self.device)))
                    position_logits = softmax(position_logits, b.info.mask_position, self.device,add_little=True)
                    # position_logits = position_logits.softmax(dim=-1)
                    old_prob_position.append(self.dist_fn(position_logits).log_prob(torch.tensor(b.act.position_id, device=self.device)))
                    spell_logits = softmax(spell_logits, b.info.mask_spell, self.device,add_little=True)
                    # spell_logits = spell_logits.softmax(dim=-1)
                    old_prob_spell.append(self.dist_fn(spell_logits).log_prob(torch.tensor(b.act.spell_id, device=self.device)))

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


    #单次输入 obs->act,mask
    def forward(self, ind,attri_stack,planes_stack,plane_glb,
                env,shooter = False,print_act = False,
                **kwargs) -> Batch:
        act_logits, targets_logits, position_logits, spell_logits, value = self.ppo_net(ind,attri_stack,planes_stack,plane_glb)
        mask_acts = env.legal_act(level=0)
        #act_id = -1
        act_logp = 0
        mask_position = np.zeros(11 * 17, dtype=bool)
        position_id = -1
        position_logp = 0
        mask_targets = np.zeros(7, dtype=bool)
        target_id = -1
        target_logp = 0
        mask_spell = np.zeros(10, dtype=bool)
        spell_id = -1
        spell_logp = 0
        value = value.item()
        if self.in_train:
            act_prob = softmax(act_logits, mask_acts, self.device)
            if print_act:
                logger.info(act_prob)
                logger.info(value)
            # if torch.sum(act_prob, dim=-1, keepdim=True).item() > 0.9:
            temp_p = self.dist_fn(act_prob)
            act_id = temp_p.sample()[0]
            act_logp = temp_p.log_prob(act_id).item()
            act_id = act_id.item()
            # else:
            #     logger.error("wrong act")
            #     logger.error(act_logits)
            #     logger.error(mask_acts)
            #     sys.exit(-1)
            #     act_prob = softmax(act_logits, mask_acts, self.device, in_train=False)
            #     act_id = torch.argmax(act_prob, dim=-1)[0].item()
            if act_id == action_type.move.value:
                mask_position = env.legal_act(level=1, act_id=act_id)
                position_prob = softmax(position_logits, mask_position, self.device)
                # if torch.sum(position_prob,dim=-1,keepdim=True).item() > 0.9:
                temp_p = self.dist_fn(position_prob)
                position_id = temp_p.sample()[0]
                position_logp = temp_p.log_prob(position_id).item()
                position_id = position_id.item()
                # else:
                #     logger.error("wrong move")
                #     logger.error(position_logits)
                #     logger.error(mask_position)
                #     sys.exit(-1)
                #     position_prob = softmax(position_logits, mask_position, self.device, in_train=False)
                #     position_id = torch.argmax(position_prob, dim=-1)[0].item()
            elif act_id == action_type.attack.value:
                mask_targets = env.legal_act(level=1, act_id=act_id)
                targets_prob = softmax(targets_logits, mask_targets, self.device)
                # if torch.sum(targets_prob,dim=-1,keepdim=True).item() > 0.9:
                temp_p = self.dist_fn(targets_prob)
                target_id = temp_p.sample()[0]
                target_logp = temp_p.log_prob(target_id).item()
                target_id = target_id.item()
                # else:
                #     logger.error("wrong attack target")
                #     logger.error(targets_prob)
                #     logger.error(mask_targets)
                #     sys.exit(-1)
                #     targets_prob = softmax(targets_logits, mask_targets, self.device, in_train=False)
                #     target_id = torch.argmax(targets_prob, dim=-1)[0].item()
                if not shooter:
                    mask_position = env.legal_act(level=2, act_id=act_id, target_id=target_id)
                    position_prob = softmax(position_logits, mask_position, self.device)
                    # if torch.sum(position_prob,dim=-1,keepdim=True).item() > 0.9:
                    temp_p = self.dist_fn(position_prob)
                    position_id = temp_p.sample()[0]
                    position_logp = temp_p.log_prob(position_id).item()
                    position_id = position_id.item()
                    # else:
                    #     logger.error("wrong attack move")
                    #     logger.error(position_logits)
                    #     logger.error(mask_position)
                    #     sys.exit(-1)
                    #     position_prob = softmax(position_logits, mask_position, self.device,in_train=False)
                    #     position_id = torch.argmax(position_prob, dim=-1)[0].item()
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
                'mask_acts': mask_acts, 'mask_spell': mask_spell, 'mask_targets': mask_targets,'mask_position': mask_position,
                'act_logp':act_logp,'target_logp':target_logp,'position_logp':position_logp,'spell_logp':spell_logp,
                'value': value}

    #@profile
    #TODO 5 mcts结构有何优势
    #批量输入训练
    def learn(self, batch, batch_size=None, repeat=1, **kwargs):
        pcount = 0
        for _ in range(repeat):
            for b in batch.split(batch_size,shuffle=True):
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
                old_prob_act = torch.tensor(b.policy.logps.act_logp, device=self.device)
                old_prob_target = torch.tensor(b.policy.logps.target_logp, device=self.device)
                old_prob_position = torch.tensor(b.policy.logps.position_logp, device=self.device)
                old_prob_spell = torch.tensor(b.policy.logps.spell_logp, device=self.device)
                adv = torch.tensor(b.adv, device=self.device)
                returns = torch.tensor(b.returns, device=self.device)
                ratio = (prob_act + prob_target + prob_position + prob_spell
                         - old_prob_act - old_prob_target - old_prob_position - old_prob_spell).exp().float()
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
                vf_loss = .5 * (returns - value).pow(2).mean()
                # vf_losses.append(vf_loss.item())
                e_loss = dist_act.entropy().mean() + dist_position.entropy().mean() + dist_targets.entropy().mean() + dist_spell.entropy().mean()
                # ent_losses.append(e_loss.item())
                #TODO 8 entropy工作原理
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
r_cum = 0
def record_sar(buffer,operator,battle,acting_stack,battle_action, obs, acts, mask, logps, value,done,killed_dealt, killed_get):
    global r_cum
    """compute reward"""
    reward = 0.
    tmp = 0.
    bl = len(buffer)
    if battle_action.type == action_type.attack:
        reward = killed_dealt * battle_action.target.ai_value / battle.ai_value[
            ~acting_stack.side] - killed_get * acting_stack.ai_value / battle.ai_value[acting_stack.side]
        if acting_stack.by_AI != operator:
            reward = -reward
    if done:
        if battle.by_AI[battle.get_winner()] == operator:  # winner is operator
            reward = 1. + reward
            win = True
    reward *= 5
    if done or acting_stack.by_AI == operator:
        #TODO 初始分兵
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
#TODO 强化学习解决旅行者问题 凸包围问题
global_buffer = ReplayBuffer(200,ignore_obs_next=True)
def collect_eps(agent,file,n_step = 200,print_act = False):
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
            if not had_acted:  # clear reward cumullated by hand craft AI
                for st in battle.stacks:
                    battle.ai_value[st.side] += st.ai_value * st.amount
                had_acted = True
        damage_dealt, damage_get, killed_dealt, killed_get = battle.doAction(battle_action)
        battle.checkNewRound()
        #battle.curStack had updated
        done = battle.check_battle_end()
        #buffer sar
        if had_acted:
            record_sar(global_buffer,2,battle,acting_stack,battle_action, obs, acts, mask, logps, value,done,killed_dealt, killed_get)
        #next act
        if done:
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


def hook_me(grad):
    print(grad.sum())
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
#@profile
def start_train():
    lrate = 0.001
    sample_num = 20
    # 初始化 agent
    actor_critic = H3_net(dev)
    optim = torch.optim.Adam(actor_critic.parameters(), lr=lrate)
    actor_critic.spells.weight.register_hook(hook_me)
    dist = torch.distributions.Categorical
    agent = H3_policy(actor_critic,optim,dist,device=dev,gae_lambda=0.95)
    # agent.load_state_dict(torch.load("model_param.pkl"))
    count = 0
    f_max = 3
    five_done = 3
    ok = five_done
    expert = load_episo("ENV/episode")
    #TODO 7 SAC算法
    while True:
        agent.eval()
        agent.in_train = True
        bats = []
        agent.process_gae(expert,single_batch=False)
        bats.append(expert)
        for ii in range(sample_num):
            fn = random.randint(0,f_max)
            file = f'ENV/battles/{fn}.json'
            print_act = False
            # if ii < 3 :
            #     print_act = True
            #     logger.info(f"------------------------{fn}.json")
            win,batch_data = collect_eps(agent,file,print_act=print_act)
            agent.process_gae(batch_data)
            bats.append(batch_data)
            if win and ii < 50:
                logger.info(f"win {fn}")
        batch_data = Batch.cat(bats)
        logger.info(len(batch_data.rew))
        agent.train()
        agent.in_train = True
        # batch_data = agent.process_gae(batch_data)
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
        win_rate = 0
        for ii in range(f_max + 1):
            file = f'ENV/battles/{ii}.json'
            ct = start_game_noGUI(file, agent=agent)
            logger.info(f"test-{count}-{ii}.json win rate = {ct}")
            win_rate += ct
        win_rate /= (f_max + 1)
        logger.info(f"win rate at all = {win_rate}")
        if win_rate > 0.9:
            ok -= 1
            if ok == 0:
                torch.save(agent.state_dict(),"model_param.pkl")
                logger.info("model saved")
                sys.exit(1)
        else:
            ok = five_done
        if count == 500:
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
def get_act_info(battle,act):
    target_id = -1
    position_id = -1
    act_id = act.type.value
    mask_acts, mask_spell, mask_targets, mask_position = battle.get_act_masks(act)
    ind, attri_stack, planes_stack, plane_glb = battle.current_state_feature()
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
def start_game_record():
    from tianshou.data import ReplayBuffer
    from ENV import H3_battleInterface
    import pygame
    buffer = ReplayBuffer(500,ignore_obs_next=True)
    # 初始化游戏
    pygame.init()  # 初始化pygame
    pygame.display.set_caption('This is my first pyVCMI')  # 设置窗口标题
    battle = Battle(by_AI = [0,1])
    battle.load_battle("ENV/battles/3.json", shuffle_postion=False,load_ai_side=False)
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
                          killed_get)
            battle.checkNewRound()
            if done:
                logger.debug("battle end~")
                bi.running = False
                pygame.quit()
                break
            else:
                acting_stack = battle.cur_stack
                bi.next_act = acting_stack.active_stack()
        bi.renderFrame()
    data, indice = buffer.sample(0)
    dump_episo([data.obs, data.obs_next, data.act, data.rew, data.done, data.info], "ENV/episode")
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
    start_game("ENV/battles/1.json", by_AI=[2, 1],agent=agent)

def main():
    # start_game_record()
    start_train()
    # start_test()

if __name__ == '__main__':
    main()