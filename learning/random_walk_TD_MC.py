#######################################################################
# Copyright (C)                                                       #
# 2016-2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)             #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
#import numpy as np
np.set_printoptions(precision=1,suppress=True,sign=' ',linewidth=400,formatter={'float': '{: 0.7f}'.format})
# 0 is the left terminal state
# 6 is the right terminal state
# 1 ... 5 represents A ... E
VALUES = np.zeros(7)
# VALUES[1:6] = 0.5
# For convenience, we assume all rewards are 0
# and the left terminal state has value 0, the right terminal state has value 1
# This trick has been used in Gambler's Problem
# VALUES[6] = 1

# set up true state values
TRUE_VALUE = np.zeros(7)
TRUE_VALUE[1:6] = np.arange(1, 6) / 6.0
# TRUE_VALUE[6] = 1

ACTION_LEFT = 0
ACTION_RIGHT = 1

# @values: current states value, will be updated if @batch is False
# @alpha: step size
# @batch: whether to update @values
def temporal_difference(values, alpha=0.1, batch=False):
    state = 3
    trajectory = [state]
    rewards = []
    while True:
        old_state = state
        if np.random.binomial(1, 0.5) == ACTION_LEFT:
            state -= 1
        else:
            state += 1
        # Assume all rewards are 0
        reward = 0
        if state == 6:
            reward = 1.0
        trajectory.append(state)
        # TD update
        if not batch:
            values[old_state] += alpha * (reward + values[state] - values[old_state])
        rewards.append(reward)
        if state == 6 or state == 0:
            break
    return trajectory, rewards

# @values: current states value, will be updated if @batch is False
# @alpha: step size
# @batch: whether to update @values
def monte_carlo(values, alpha=0.1, batch=False):
    state = 3
    trajectory = [3]

    # if end up with left terminal state, all returns are 0
    # if end up with right terminal state, all returns are 1
    while True:
        if np.random.binomial(1, 0.5) == ACTION_LEFT:
            state -= 1
        else:
            state += 1
        trajectory.append(state)
        if state == 6:
            returns = 1.0
            break
        elif state == 0:
            returns = 0.0
            break

    if not batch:
        for state_ in trajectory[:-1]:
            # MC update
            values[state_] += alpha * (returns - values[state_])
    return trajectory, [returns] * (len(trajectory) - 1)

# Example 6.2 left
def compute_state_value():
    episodes = [0, 1, 10, 100]
    current_values = np.copy(VALUES)
    plt.figure(1)
    for i in range(episodes[-1] + 1):
        if i in episodes:
            plt.plot(current_values, label=str(i) + ' episodes')
        temporal_difference(current_values)
    plt.plot(TRUE_VALUE, label='true values')
    plt.xlabel('state')
    plt.ylabel('estimated value')
    plt.legend()

# Example 6.2 right
def rms_error():
    # Same alpha value can appear in both arrays
    td_alphas = [0.15, 0.1, 0.05]
    mc_alphas = [0.01, 0.02, 0.03, 0.04]
    episodes = 100 + 1
    runs = 100
    for i, alpha in enumerate(td_alphas + mc_alphas):
        total_errors = np.zeros(episodes)
        if i < len(td_alphas):
            method = 'TD'
            linestyle = 'solid'
        else:
            method = 'MC'
            linestyle = 'dashdot'
        for r in tqdm(range(runs)):
            errors = []
            current_values = np.copy(VALUES)
            for i in range(0, episodes):
                errors.append(np.sqrt(np.sum(np.power(TRUE_VALUE - current_values, 2)) / 5.0))
                if method == 'TD':
                    temporal_difference(current_values, alpha=alpha)
                else:
                    monte_carlo(current_values, alpha=alpha)
            total_errors += np.asarray(errors)
        total_errors /= runs
        plt.plot(total_errors, linestyle=linestyle, label=method + ', alpha = %.02f' % (alpha))
    plt.xlabel('episodes')
    plt.ylabel('RMS')
    plt.legend()

# Figure 6.2
# @method: 'TD' or 'MC'
def batch_updating(method, episodes, alpha=0.001):
    # perform 100 independent runs
    runs = 100
    total_errors_td = np.zeros(episodes)
    total_errors_mc = np.zeros(episodes)
    for r in tqdm(range(0, runs)):
        print(f"run - {r}")
        current_values_td = np.copy(VALUES)
        current_values_mc = np.copy(VALUES)
        errors_td = []
        errors_mc = []
        # track shown trajectories and reward/return sequences
        trajectories = []
        rewards = []
        for ep in range(episodes):
            trajectory_, rewards_ = temporal_difference(current_values_td, batch=True)
            trajectories.append(trajectory_)
            rewards.append(rewards_)
            batch_update_inner(trajectories, rewards, current_values_td, current_values_mc, alpha=alpha)
            # calculate rms error
            errors_td.append(np.sqrt(np.sum(np.power(current_values_td - TRUE_VALUE, 2)) / 5.0))
            errors_mc.append(np.sqrt(np.sum(np.power(current_values_mc - TRUE_VALUE, 2)) / 5.0))
        total_errors_td += np.asarray(errors_td)
        total_errors_mc += np.asarray(errors_mc)
    total_errors_td /= runs
    total_errors_mc /= runs
    return total_errors_td,total_errors_mc
def batch_update_inner(trajectories,rewards,current_values_td,current_values_mc,alpha=.001):
    ii = 0
    td_done = False
    mc_done = False
    while True:
        # keep feeding our algorithm with trajectories seen so far until state value function converges
        updates_td = np.zeros(7)
        updates_mc = np.zeros(7)
        if not td_done:
            for trajectory_, rewards_ in zip(trajectories, rewards):
                # print("new traj")
                for i in range(0, len(trajectory_) - 1):
                    updates_td[trajectory_[i]] += rewards_[i] + current_values_td[trajectory_[i + 1]] - \
                                                  current_values_td[trajectory_[i]]
                    # print("%i to %i, td diff %f " % (trajectory_[i], trajectory_[i + 1],
                    #                                  rewards_[i] + current_values_td[trajectory_[i + 1]] -
                    #                                  current_values_td[
                    #                                      trajectory_[i]]), end='. ')

            # print()
            updates_td *= alpha
            td_updates = np.sum(np.abs(updates_td))
            if td_updates < 1e-3:
                td_done = True
                print("end td ii = ", ii)
            else:
                # print("td update = ", np.sum(np.abs(td_updates / alpha), axis=-1))
                current_values_td += updates_td
        if not mc_done:
            for trajectory_, rewards_ in zip(trajectories, rewards):
                # print("new traj")
                if rewards_[-1] > 0:
                    rewards_ = [rewards_[-1]] * (len(trajectory_) - 1)
                for i in range(0, len(trajectory_) - 1):
                    updates_mc[trajectory_[i]] += rewards_[i] - current_values_mc[trajectory_[i]]
                    # print("%i to %i, mc diff %f " % (trajectory_[i], trajectory_[i + 1],
                    #                                  rewards_[i] -
                    #                                  current_values_mc[
                    #                                      trajectory_[i]]), end='. ')
            # print()
            updates_mc *= alpha
            mc_updates = np.sum(np.abs(updates_mc))
            if mc_updates < 1e-3:
                print("end mc ii = ", ii)
                mc_done = True
            else:
                # print("mc update = ", np.sum(np.abs(mc_updates / alpha), axis=-1))
                current_values_mc += updates_mc

        # print("current_values td", current_values_td)
        # print("current_values mc", current_values_mc)
        # perform batch updating
        ii += 1
        if td_done and mc_done:
            break
def example_6_2():
    plt.figure(figsize=(10, 20))
    plt.subplot(2, 1, 1)
    compute_state_value()

    plt.subplot(2, 1, 2)
    rms_error()
    plt.tight_layout()

    plt.savefig('./example_6_2.png')
    plt.close()

def figure_6_2():
    episodes = 100 + 1
    td_erros,mc_erros = batch_updating('TD', episodes,.001)
    # mc_erros = batch_updating('MC', episodes,.001)

    plt.plot(td_erros, label='TD')
    plt.plot(mc_erros, label='MC')
    plt.xlabel('episodes')
    plt.ylabel('RMS error')
    plt.legend()

    plt.savefig('./figure_6_2.png')
    plt.close()
def xx():
    # a =0.4
    # b=0.6
    # alpha = 0.3
    # for i in range(5):
    #     a += alpha*(b -a)
    #     b += alpha*(a - b)
    #     print("a=%f,b=%f"%(a,b))
    x = [1, 3, 7, 8, 12, 14, 17, 25, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]
    y = [3, 5, 6, 10, 13, 23, 24, 27, 30, 28, 22, 12, 15, 16, 11, 8, 7, 15, 25]
    plt.plot(x, y, label='y=f(x)')
    plt.xlabel('key') # xlabel 方法指定 x 轴显示的名字
    plt.ylabel('value') # ylabel 方法指定 y 轴显示的名字
    plt.title('k-v trend')
    plt.legend() # legend 是在图区显示label，即上面 .plot()方法中label参数的值
    plt.show()


if __name__ == '__main__':
    # example_6_2()
    figure_6_2()
    # xx()
    # traj = np.array([3,4,5,4,3,4,5,4,3,4,5,4,3,4,5,4,3,4,5,4,3,4,5,4,3,4,5,4,3,4,5,4,3,4,5,4,3,4,5,4,3,4,5,4,3,4,5,4,5,6])
    # reward = np.zeros((len(traj)-1,),dtype=float)
    # reward[-1] = 1.
    #
    # current_values_td = np.copy(VALUES)
    # current_values_mc = np.copy(VALUES)
    # batch_update_inner([traj],[reward],current_values_td,current_values_mc)