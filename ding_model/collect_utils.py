
def process_standard_g(data: list,logger):
    r"""
    reward = [2,3,4,5]
    prev_reward = reward[:-1] = [0,2,3,4]
    r = reward - prev_reward = [2,1,1,1]
    g = reward[-1] - prev_reward = [5,3,2,1]
    """
    last_done = data[-1]['real_done']
    last_done = int(last_done)
    if last_done < len(data) - 1:
        assert data[last_done]['real_done']
        data = data[:last_done + 1]
        data[-1]['done'] = True
        data[-1]['real_done'] = last_done
        logger.info(f"end_reward={data[-1]['reward']}")
    prev_data = [None] + data[:-1]
    end_reward = 0
    for step, prev_step in zip(data[::-1], prev_data[::-1]):
        # assert step['obs']['action_mask'][step['action']].item() == 1,f"action id = {step['action'].item()}"
        if step["real_done"]:
            end_reward += step['reward']
        if prev_step:
            step['g'] = end_reward - prev_step['reward'] * (1 - int(prev_step["real_done"]))
        else:
            step['g'] = end_reward
        step['adv'] = step['g'] - step['value']
    return data