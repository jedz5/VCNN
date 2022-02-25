
def process_standard_g(data: list,logger=None):
    r"""
    reward = [2,3,4,5]
    prev_reward = reward[:-1] = [0,2,3,4]
    r = reward - prev_reward = [2,1,1,1]
    g = reward[-1] - prev_reward = [5,3,2,1]
    """
    # last_done = data[-1]['real_done']
    # last_done = int(last_done)
    # if last_done < len(data) - 1:
    #     assert data[last_done]['real_done']
    #     data = data[:last_done + 1]
    #     data[-1]['done'] = True
    #     data[-1]['real_done'] = last_done
    #     logger.info(f"end_reward={data[-1]['reward']}")
    win = False
    prev_data = [None] + data[:-1]
    end_reward = 0
    for step, prev_step in zip(data[::-1], prev_data[::-1]):
        # assert step['obs']['action_mask'][step['action']].item() == 1,f"action id = {step['action'].item()}"
        if step["real_done"]:
            if win:
                logger.info(f"end_reward={data[-1]['reward']}")
                end_reward = max(end_reward,0) #最后一场的负值不能传递到前面场次
                logger.info(f"current_reward={end_reward}")
            end_reward += step['reward']
            win = True
        if prev_step:
            step['g'] = end_reward - prev_step['reward'] * (1 - int(prev_step["real_done"]))
        else:
            step['g'] = end_reward
        step['adv'] = step['g'] - step['value']
    return data
bFieldWidth = 17
bFieldSize = 15 * 11
def indexToAction_simple(move):
    move = int(move)
    if (move < 0):
        print('wrong move {}'.format(move))
        exit(-1)
    if (move == 0):
        return "w"
    elif (move == 1):
        return "d"
    elif ((move - 2) >= 0 and (move - 2) < bFieldSize):
        y = (move - 2) // (bFieldWidth - 2)
        x = (move - 2) % (bFieldWidth - 2) + 1
        return f"m({y},{x})"
    elif ((move - 2 - bFieldSize) >= 0 and (move - 2 - bFieldSize) < 14):
        enemy_id = move - 2 - bFieldSize
        return f"sh({enemy_id})"
    elif ((move - 2 - bFieldSize - 14) >= 0):
        direction = (move - 2 - bFieldSize - 14) % 6
        enemy_id = (move - 2 - bFieldSize - 14) // 6
        return f"att({enemy_id},d{direction})"
    else:
        print('wrong move {}'.format(move))
        exit(-1)