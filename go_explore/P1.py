from ENV.H3_battle import *




def start_game():
    battle = Battle(by_AI = [0,1])
    battle.load_battle("ENV/battles/6.json", shuffle_postion=False,load_ai_side=False)
    # battle.loadFile("ENV/debug.json",load_ai_side=False)
    battle.checkNewRound()
    # 事件循环(main loop)
    while True:
        do_act = bi.handleEvents()
        if do_act:
            battle.doAction(bi.next_act)
            battle.checkNewRound()
            done = battle.check_battle_end()
            if done:
                logger.debug("battle end~")
                break
            else:
                bi.next_act = battle.cur_stack.active_stack()
        bi.renderFrame()

if __name__ == '__main__':
    start_game()