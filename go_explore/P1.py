from dataclasses import dataclass
from collections import defaultdict
from ENV.H3_battle import *
from VCbattle import BHex
import pygame

# import loky
from contextlib import contextmanager
import multiprocessing

from ENV.H3_battleInterface import *
@dataclass
class cell:
    __slots__ = ['restore', 'score']
    restore:tuple
    # choose_time:int
    # seen_time:int
    score:int
@dataclass
class cell_key:
    __slots__ = ['states', 'round']
    states:tuple
    round:int
class LPool:
    def __init__(self, n_cpus, maxtasksperchild=100):
        self.pool = loky.get_reusable_executor(n_cpus, timeout=100)

    def map(self, f, r):
        return self.pool.map(f, r)
@contextmanager
def use_seed(seed):
    # Save all the states
    python_state = random.getstate()
    np_state = np.random.get_state()

    # Seed all the rngs (note: adding different values to the seeds
    # in case the same underlying RNG is used by all and in case
    # that could be a problem. Probably not necessary)
    random.seed(seed)
    np.random.seed(seed + 1)

    # Yield control!
    yield

    # Reset the rng states
    random.setstate(python_state)
    np.random.set_state(np_state)
def run_f_seeded(args):
    f, seed, args = args
    with use_seed(seed):
        return f(args)


class SeedPoolWrap:
    def __init__(self, pool):
        self.pool = pool

    def map(self, f, r):
        return self.pool.map(run_f_seeded, [(f, random.randint(0, 2**32 - 10), e) for e in r])


def seed_pool_wrapper(pool_class):
    def f(*args, **kwargs):
        return SeedPoolWrap(pool_class(*args, **kwargs))
    return f
go_explorer = defaultdict(cell)
# POOL = seed_pool_wrapper(LPool)(multiprocessing.cpu_count() * 2) # multiprocessing.get_context("spawn").Pool
# POOL = pool_class(multiprocessing.cpu_count() * 2)
class explore_agent:
    def __init__(self,side):
        self.side = side
    def choose_action(self,in_battle:Battle,ret_obs = False,print_act=False,action_known=None):
        cur = in_battle.cur_stack
        assert self.side == cur.side,"wrong side?"
        legal_act = in_battle.legal_act(level=0)
        act = np.random.choice(list(range(len(legal_act))),p=legal_act/legal_act.sum())
        act = action_type(act)
        if act == action_type.wait:
            return BAction(type=action_type.wait)
        elif act == action_type.defend:
            return BAction(type=action_type.defend)
        elif act == action_type.move:
            legal_move = in_battle.legal_act(level=1,act_id=action_type.move.value)
            dest = np.random.choice(list(range(len(legal_move))), p=legal_move / legal_move.sum())
            return BAction(type=action_type.move,dest=BHex(dest % Battle.bFieldWidth, int(dest / Battle.bFieldWidth)))
        elif act == action_type.attack:
            legal_target = in_battle.legal_act(level=1,act_id=action_type.attack.value)
            target_id = np.random.choice(list(range(len(legal_target))), p=legal_target / legal_target.sum())
            t = in_battle.stackQueue[target_id]
            if cur.can_shoot():
                return BAction(type=action_type.attack,target = t)
            else:
                legal_dest = in_battle.legal_act(level=2,act_id=action_type.attack.value,target_id=target_id)
                dest = np.random.choice(list(range(len(legal_dest))), p=legal_dest / legal_dest.sum())
                return BAction(type=action_type.attack,target = t,dest=BHex(dest % Battle.bFieldWidth, int(dest / Battle.bFieldWidth)))
        else:
            logger.error("not implemented yet!")
            sys.exit()
def collect_eps(battle:Battle=None,file=None,n_step = 100,print_act = False):
    if not battle:
        battle = Battle()
        battle.load_battle(file)
    battle.checkNewRound()
    traj = []
    for ii in range(n_step):
        #battle.curStack had updated
        acting_stack = battle.cur_stack
        #buffer sar
        state = battle.state_represent()
        if acting_stack.by_AI == 2:
            go_explorer[state] = cell((battle.current_state_feature(curriculum=True),battle.round),0)
        #next act
        if acting_stack.by_AI == 2:
            battle_action = acting_stack.active_stack(ret_obs=False, print_act=print_act)
        else:
            battle_action = acting_stack.active_stack()
        traj.append(f"{acting_stack.name}({acting_stack.amount})\t({acting_stack.y},{acting_stack.x})\t{battle_action}")
        battle.doAction(battle_action)
        battle.checkNewRound()
        done = battle.check_battle_end()
        if done:
            win = (battle.by_AI[battle.get_winner()] == 2)
            if win:
                go_explorer[state].score = 1
                print("agent win!!!")
                for tj in traj:
                    print(tj)
                sys.exit()
            break
    return
def start_explore():
    # pygame.init()  # 初始化pygame
    # pygame.display.set_caption('This is my first pyVCMI')  # 设置窗口标题
    # bi = BattleInterface()
    agent = explore_agent(side=0)
    for step in range(10):
        arena = Battle(agent=agent)
        arena.load_battle("ENV/battles/6.json")
        collect_eps(battle=arena)
    for i in range(1000):
        print(f"iter:{i}")
        print(len(go_explorer.keys()))
        k = go_explorer.keys()
        rk = range(len(k))
        lk = list(k)
        idx = np.random.choice(rk,100)
        for id in idx:
            cell = go_explorer[lk[id]]
            arena = Battle(agent=agent)
            arena.load_battle(file=cell.restore)
            # if np.random.randint(0,100) < 1:
            if False:
                start_game_gui(battle_int=bi,battle=arena)
            else:
                collect_eps(battle=arena)
def simple_explore():
    arena = Battle()
    arena.load_battle("ENV/battles/5.json")
    arena.checkNewRound()
    start = arena.current_state_feature(curriculum=True)
    for i in range(20000):
        print(f"iter:{i}")
        traj = [e for e in POOL.map(simple_run, [start]*100)]
        print(traj)
def simple_run(start):
    agent = explore_agent(side=0)
    arena = Battle(agent=agent)
    for i in range(100):
        arena.load_battle((start,0))
        arena.checkNewRound()
        traj = []
        for ii in range(50):
            # battle.curStack had updated
            acting_stack = arena.cur_stack
            # buffer sar
            # next act
            if acting_stack.by_AI == 2:
                battle_action = acting_stack.active_stack()
            else:
                battle_action = acting_stack.active_stack()

            traj.append(f"{acting_stack.name}({acting_stack.amount})\t({acting_stack.y},{acting_stack.x})\t{battle_action}")
            arena.doAction(battle_action)
            arena.checkNewRound()
            done = arena.check_battle_end()
            if done:
                win = (arena.by_AI[arena.get_winner()] == 2)
                if win:
                    print("agent win!!!")
                    for tj in traj:
                        print(tj)
                    return traj
            break

    return 0
M = 0
if __name__ == '__main__':
    start_explore()