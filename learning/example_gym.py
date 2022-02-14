import gym
import sys
import keyboard
# from gym.envs.atari import AtariEnv

act = 0

r''' v0 自带sticky=0.25、frameskip=2~5,deterministic下frameskip=4，noframeskip下frameskip=1，v4 sticky=0  '''
def gym_ui():
    env = gym.make("ALE/MontezumaRevenge-v5", render_mode='human') #MontezumaRevenge
    # env = AtariEnv("pong",frameskip=1,render_mode='human',repeat_action_probability=0.8)
    env.reset()

    def abc(event):
        global act
        if event.event_type == 'down':
            if event.name == 'a':
                act = 4
            elif event.name == 'd':
                act = 3
            elif event.name == 'w':
                act = 3 #2
            elif event.name == 's':
                act = 4 #5
            else:
                if act == 4:
                    act = 12
                elif act == 3:
                    act = 11
                else:
                    act = 1
        if event.event_type == 'up':
            act = 0

    keyboard.hook(abc)
    while True:
        env.render(mode='human')
        if act != 0:
            print(act)
        env.step(act)  # env.action_space.sample()

    env.close()

if __name__ == '__main__':
    gym_ui()
    # from gym import envs
    #
    # print(envs.registry.all())