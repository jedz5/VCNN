from typing import List, Optional
import gym
import copy
import numpy as np

import sys

from ENV.H3_battle import Battle
from ding.envs import BaseEnvTimestep, BaseEnvInfo, DingEnvWrapper
from ding.envs.common.env_element import EnvElementInfo
from ding.torch_utils import to_ndarray


def check_done(battle:Battle):
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
# def cumulate_reward(buffer,start,end):
#     assert end != start
#     assert buffer.done[end - 1] > 0
#     tmp = None
#     if buffer.rew[end - 1] < 0:
#         tmp = buffer.rew[end - 1]
#         buffer.rew[end - 1] = 0
#     if end > start:
#         batch_rew = buffer.rew[start:end]
#         batch_done = buffer.done[start:end]
#         acc_rewards = np.add.accumulate(batch_rew[batch_done > 0][::-1])
#         batch_rew[batch_done > 0] = acc_rewards[::-1]
#     else:
#         idx = list(range(start, len(buffer))) + list(range(end))
#         batch_rew = buffer.rew[idx]
#         batch_done = buffer.done[idx]
#         acc_rewards = np.add.accumulate(batch_rew[batch_done > 0][::-1])
#         batch_rew[batch_done > 0] = acc_rewards[::-1]
#         buffer.rew[idx] = batch_rew
#     if tmp:
#         buffer.rew[end - 1] = tmp
reward_def = 1.
def compute_reward(battle: Battle):
    att_HP = [st.amount_base * st.health for st in battle.attacker_stacks]
    def_HP = [st.amount_base * st.health for st in battle.defender_stacks]
    att_HP_left = [(st.amount - 1) * st.health + st.first_HP_Left for st in battle.attacker_stacks if
                   st.amount > 0]
    def_HP_left = [(st.amount - 1) * st.health + st.first_HP_Left for st in battle.defender_stacks if
                   st.amount > 0]
    reward = reward_def * (sum(att_HP_left) / sum(att_HP) - sum(def_HP_left) / sum(def_HP))
    return reward
class GymH3EnvWrapper:
    def __init__(self,battle:Battle):
        self.battle = battle
        self.orig_attacker_stacks = self.battle.attacker_stacks
    def reset(self,**param):
        if not ('continue_round' in param and param['continue_round']):
            self.battle.attacker_stacks = self.orig_attacker_stacks
        self.battle.split_army(**param)
        self.battle.checkNewRound()
        end = self.battle.check_battle_end()
        while(not end and self.battle.cur_stack.by_AI == 1 ):
            act = self.battle.cur_stack.active_stack()
            self.battle.doAction(act)
            self.battle.checkNewRound()
            end = self.battle.check_battle_end()
        if end:
            print("died at first?")
            sys.exit(-1)
        ind, attri_stack, planes_stack, plane_glb = self.battle.current_state_feature()
        action_mask = self.battle.act_mask_flatten()
        obs = {'ind': ind, 'attri_stack': attri_stack, 'planes_stack': planes_stack, 'plane_glb': plane_glb,'action_mask':action_mask}
        return obs
    def step(self,action):
        act = self.battle.indexToAction(action)
        self.battle.doAction(act)
        self.battle.checkNewRound()
        end = self.battle.check_battle_end()
        while (not end and self.battle.cur_stack.by_AI == 1):
            act = self.battle.cur_stack.active_stack()
            self.battle.doAction(act)
            self.battle.checkNewRound()
            end = self.battle.check_battle_end()
        ind, attri_stack, planes_stack, plane_glb = self.battle.current_state_feature()
        action_mask = self.battle.act_mask_flatten()
        obs = {'ind': ind, 'attri_stack': attri_stack, 'planes_stack': planes_stack, 'plane_glb': plane_glb,'action_mask':action_mask}
        # battle.curStack had updated
        # check done
        done, win = check_done(self.battle)
        info = {}
        rew = 0.
        if done:
            if win:
                rew = compute_reward(self.battle)
            else:
                rew = -reward_def
        else:
            rew = compute_reward(self.battle)
        return obs,rew,done,info
    def close(self) -> None:
        self.battle.clear()
    # override
    # def seed(self, seed: int, dynamic_seed: bool = True) -> None:
    #     self._seed = seed
    #     self._dynamic_seed = dynamic_seed
    #     np.random.seed(self._seed)
class DingH3Env(DingEnvWrapper):

    def __init__(self, env: GymH3EnvWrapper, cfg: dict = None) -> None:
        self._cfg = cfg
        if self._cfg is None:
            self._cfg = dict()
        self._env = env
        self.outter_round = 0
        self.last_done_index = -1
        self.step_count = 0

    # override
    def reset(self) -> None:
        # if hasattr(self, '_seed') and hasattr(self, '_dynamic_seed') and self._dynamic_seed:
        #     np_seed = 100 * np.random.randint(1, 1000)
        #     self._env.seed(self._seed + np_seed)
        # elif hasattr(self, '_seed'):
        #     self._env.seed(self._seed)
        self.outter_round = 0
        self.last_done_index = -1
        self.step_count = 0
        obs = self._env.reset(continue_round=False)
        # obs = to_ndarray(obs).astype(np.float32)
        self._final_eval_reward = 0.0
        self._action_type = self._cfg.get('action_type', 'scalar')
        return obs

    # override
    def close(self) -> None:
        self._env.close()

    # override
    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)

    # override
    def step(self, action: np.ndarray) -> BaseEnvTimestep:
        assert isinstance(action, np.ndarray), type(action)
        if action.shape == (1, ) and self._action_type == 'scalar':
            action = action.squeeze()
        obs, rew, done, info = self._env.step(action)
        info['real_done'] = done
        if done:
            self._final_eval_reward += rew
            if rew > 0:
                if self.outter_round < 10:
                    self.last_done_index = self.step_count
                    print(f"here round={self.outter_round}")
                    done = False
                    obs = self._env.reset()
            else:
                if self.outter_round > 0:
                    self._final_eval_reward -= rew
                    print(f"last done index={self.last_done_index}")
                    rew = -self.last_done_index
            info['final_eval_reward'] = self._final_eval_reward
            self.outter_round += 1
        self.step_count += 1
        rew = to_ndarray([rew])  # wrapped to be transferred to a Tensor with shape (1,)
        return BaseEnvTimestep(obs, rew, done, info)

    def info(self) -> BaseEnvInfo:
        obs_space = self._env.observation_space
        act_space = self._env.action_space
        return BaseEnvInfo(
            agent_num=1,
            obs_space=EnvElementInfo(
                shape=obs_space.shape,
                value={
                    'min': obs_space.low,
                    'max': obs_space.high,
                    'dtype': np.float32
                },
            ),
            act_space=EnvElementInfo(
                shape=(act_space.n, ),
                value={
                    'min': 0,
                    'max': act_space.n,
                    'dtype': np.float32
                },
            ),
            rew_space=EnvElementInfo(
                shape=1,
                value={
                    'min': -1,
                    'max': 1,
                    'dtype': np.float32
                },
            ),
            use_wrappers=None
        )

    def __repr__(self) -> str:
        return "DI-engine H3 Env({})".format(self._cfg.env_id)

    @staticmethod
    def create_collector_env_cfg(cfg: dict) -> List[dict]:
        actor_env_num = cfg.pop('collector_env_num')
        cfg = copy.deepcopy(cfg)
        cfg.is_train = True
        return [cfg for _ in range(actor_env_num)]

    @staticmethod
    def create_evaluator_env_cfg(cfg: dict) -> List[dict]:
        evaluator_env_num = cfg.pop('evaluator_env_num')
        cfg = copy.deepcopy(cfg)
        cfg.is_train = False
        return [cfg for _ in range(evaluator_env_num)]

    def enable_save_replay(self, replay_path: Optional[str] = None) -> None:
        if replay_path is None:
            replay_path = './video'
        self._replay_path = replay_path
        # this function can lead to the meaningless result
        # disable_gym_view_window()
        self._env = gym.wrappers.Monitor(
            self._env, self._replay_path, video_callable=lambda episode_id: True, force=True
        )