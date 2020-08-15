import pygame  # 导入pygame库
from ENV.H3_battle import *
import math
# import VCCC.x64.Release.VCbattle  as vb
from VCbattle import BHex
COMBAT_BLOCKED, COMBAT_MOVE, COMBAT_FLY, COMBAT_SHOOT,COMBAT_HERO, COMBAT_QUERY, COMBAT_POINTER = range(7)
COMBAT_SHOOT_PENALTY,COMBAT_SHOOT_CATAPULT, COMBAT_HEAL,COMBAT_SACRIFICE, COMBAT_TELEPORT = range(15,20)

class log_with_gui(object):
    def __init__(self,std_logger):
        self.logger = std_logger
        self.log_text = []
    def info(self,text,to_gui = False):
        # pass
        self.logger.info(text)
        if(to_gui):
            self.log_text.append(text)
    def debug(self,text,to_gui = False):
        # pass
        self.logger.debug(text)
        if(to_gui):
            self.log_text.append(text)
    def error(self,text,to_gui = False):
        # pass
        self.logger.error(text)
        if(to_gui):
            self.log_text.append(text)

logger = log_with_gui(get_logger()[1])
set_logger(True,logger)
class BPoint:
    def __init__(self,x,y):
        self.x = x
        self.y = y

class CClickableHex:
    def __init__(self, hex_i=0, hex_j=0, pixels_x=0, pixels_y=0):
        self.hex_i = hex_i
        self.hex_j = hex_j
        self.pixels_x = pixels_x
        self.pixels_y = pixels_y
        self.hovered = False

    def mouseMoved(self):
        self.hovered = True
    @classmethod
    def XYtoIJ(self,mouse_x,mouse_y):
        i = (int)((mouse_y - 86) / 42)
        j = (int)((mouse_x - (14 + (22 if i % 2 == 0 else 0))) / 44)
        return i,j
    @classmethod
    def getXYUnitAnim(self,hex,stack):
        ret = BPoint(-500,-500)
        basePos = BPoint(-190,-139) # position of creature in topleft corner
        imageShiftX = 30 # X offset to base pos for facing right stacks, negative for facing left
        ret.x = basePos.x + 22 * ((hex.y + 1) % 2) + 44 * hex.x;
        ret.y = basePos.y + 42 * hex.y;
        if not stack.side:
            ret.x += imageShiftX
        else:
            ret.x -= imageShiftX
        return ret
        # if stack.isWide:
    def getHexXY(self):
        if not self.pixels_x :
            self.pixels_x = 14 + (22 if self.hex_i % 2 == 0 else 0) + 44 * self.hex_j
            self.pixels_y = 86 + 42 * self.hex_i
        return self.pixels_x,self.pixels_y
class BattleInterface:
    def __init__(self, battle_engine = None):
        # 定义窗口的分辨率
        self.SCREEN_WIDTH = 1000
        self.SCREEN_HEIGHT = 600
        self.BFIELD_WIDTH = 17
        self.BFIELD_HEIGHT = 11
        self.running = True
        self.current_hex = CClickableHex()
        self.hoveredStack = None
        self.transColor = pygame.Color(255, 0, 255)
        self.stimgs = {}
        self.stimgs_dead = {}
        self.cursor = None
        self.cursor_pos = (0,0)
        self.cursor_attack = [None] * 6
        self.cursor_move = [None] * 5
        self.cursor_shoot = [None] * 2
        self.loadIMGs()
        if battle_engine:
            self.init_battle(battle_engine)
        else:
            self.battle_engine = None
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 11)
        self.font.set_bold(True)
        self.next_act = None
        self.act = None
        self.dump_dir = "ENV/battles"
        pygame.mouse.set_visible(False)
    def loadIMGs(self):
        self.screen = pygame.display.set_mode([self.SCREEN_WIDTH, self.SCREEN_HEIGHT])  # 初始化一个用于显示的窗口
        background = pygame.image.load("ENV/imgs/bgrd.bmp")
        self.background = background
        self.hex_shader = pygame.image.load("ENV/imgs/CCellShd_gray.bmp")
        self.hex_shader.set_colorkey(self.transColor)
        self.hex_shader.set_alpha(100)
        self.shader_cur_stack = pygame.image.load("ENV/imgs/CCellShd_green.bmp")
        self.shader_cur_stack.set_colorkey(self.transColor)
        self.shader_cur_target = pygame.image.load("ENV/imgs/CCellShd_red.bmp")
        self.shader_cur_target.set_colorkey(self.transColor)
        self.amout_backgrd = pygame.image.load("ENV/imgs/CmNumWin_purple.bmp").convert_alpha()
        self.amout_backgrd_enemy = pygame.image.load("ENV/imgs/CmNumWin_blue.bmp").convert_alpha()

        for idx in range(6):
            img = pygame.image.load("ENV/imgs/cursor/attack/" + str(idx) + ".bmp")#.convert_alpha()
            # img.set_colorkey((0, 255, 255))
            self.cursor_attack[idx] = img
        for idx in range(5):
            img = pygame.image.load("ENV/imgs/cursor/move/" + str(idx) + ".bmp")#.convert_alpha()
            # img.set_colorkey((0, 255, 255))
            self.cursor_move[idx] = img
        for idx in range(2):
            img = pygame.image.load("ENV/imgs/cursor/shoot/" + str(idx) + ".bmp")#.convert_alpha()
            # img.set_colorkey((0, 255, 255))
            self.cursor_shoot[idx] = img
        self.cursor = self.cursor_move[0]
    def init_battle(self,battle):
        self.running = True
        battle.bat_interface = self
        self.battle_engine = battle
        for st in battle.stacks:
            if st.name not in self.stimgs:
                img = pygame.image.load("ENV/imgs/creatures/"+st.name+".bmp").convert_alpha()
                imgback = pygame.Surface(img.get_size())
                imgback.blit(img, (0, 0))
                imgback.set_colorkey((16,16,16))
                self.stimgs[st.name] = imgback
                #dead
                img = pygame.image.load("ENV/imgs/creatures/dead/" + st.name + ".bmp").convert_alpha()
                imgback = pygame.Surface(img.get_size())
                imgback.blit(img, (0, 0))
                imgback.set_colorkey((16, 16, 16))
                self.stimgs_dead[st.name] = imgback
        # for oi in self.battle_engine.obsinfo:
        #     if oi.imname not in self.stimgs:
        #         img = pygame.image.load("ENV/imgs/obstacles/" + oi.imname + ".bmp").convert_alpha()
        #         imgback = pygame.Surface(img.get_size())
        #         imgback.blit(img, (0, 0))
        #         if oi.isabs:
        #             imgback.set_colorkey((0, 255, 255))
        #         else:
        #             imgback.set_colorkey((0, 0, 0))
        #         self.stimgs[oi.imname] = imgback
    def renderFrame(self):
        if self.running:
            self.clock.tick(60)
            self.update()
            pygame.display.update()

    def showBackground(self):
        self.screen.fill((0,0,0))
        self.screen.blit(self.background, (0, 0))
        self.screen.blit(self.hex_shader, (self.current_hex.pixels_x, self.current_hex.pixels_y))

    def showBattlefieldObjects(self):
        ##########
        ##########
        stacks = self.battle_engine.stacks
        #show active stack
        curStackRange = self.battle_engine.cur_stack.get_global_state()
        for i in range(self.battle_engine.bFieldHeight):
            for j in range(self.battle_engine.bFieldWidth):
                if 0 <= curStackRange[i][j] < 50:
                    self.screen.blit(self.hex_shader, CClickableHex(i,j).getHexXY())
        self.screen.blit(self.shader_cur_stack, CClickableHex(self.battle_engine.cur_stack.y, self.battle_engine.cur_stack.x).getHexXY())
        if self.battle_engine.cur_stack.by_AI > 0:
            #stack target
            if (self.next_act.type == action_type.wait):
                pass
            elif (self.next_act.type == action_type.defend):
                pass
            elif (self.next_act.type == action_type.move):
                self.screen.blit(self.shader_cur_target,
                                 CClickableHex(self.next_act.dest.y, self.next_act.dest.x).getHexXY())
            elif (self.next_act.type == action_type.attack):
                self.screen.blit(self.shader_cur_target,
                                 CClickableHex(self.next_act.dest.y, self.next_act.dest.x).getHexXY())
                self.screen.blit(self.shader_cur_target,
                                 CClickableHex(self.next_act.target.y, self.next_act.target.x).getHexXY())
            if (self.next_act.type == action_type.shoot):
                self.screen.blit(self.shader_cur_target,
                                 CClickableHex(self.next_act.target.y, self.next_act.target.x).getHexXY())
        # show obstacles
        for oi in self.battle_engine.obsinfo:
            img = self.stimgs[oi.imname]
            x, y = self.getObstaclePosition(img, oi)
            self.screen.blit(img, (x, y))

        # hover on stack

        if self.hoveredStack:
            hoveredRange = self.hoveredStack.get_global_state()
            for i in range(self.battle_engine.bFieldHeight):
                for j in range(self.battle_engine.bFieldWidth):
                    if 0 <= hoveredRange[i][j] < 50:
                        self.screen.blit(self.hex_shader, CClickableHex(i,j).getHexXY())


        # show stacks
        for stack in stacks:
                coord = CClickableHex.getXYUnitAnim(BHex(stack.x,stack.y),stack)
                if (stack.is_alive()):
                    img = self.stimgs[stack.name]
                    img = pygame.transform.flip(img, True, False) if stack.side else img
                else:
                    img = self.stimgs_dead[stack.name]
                    img = pygame.transform.flip(img, True, False) if stack.side else img
                self.screen.blit(img,(coord.x,coord.y))

        # show stack amounts
        for stack in stacks:
            coord = CClickableHex.getXYUnitAnim(BHex(stack.x, stack.y), stack)
            if (stack.is_alive()):
                if(stack.side == 0):
                    amount_bgrd = copy.copy(self.amout_backgrd)
                else:
                    amount_bgrd = copy.copy(self.amout_backgrd_enemy)

                retaliate = "" if stack.had_retaliated else "r"
                waited = "w" if stack.had_waited and not stack.had_moved else ""
                moved = "m" if stack.had_moved else ""
                text_surface = self.font.render(u"{}{}{}{}              #".format(stack.amount,retaliate,moved,waited), True, (255, 255, 255))
                amount_bgrd.blit(text_surface, (2, -2))
                xadd = 220 - (44 if stack.side else -22)
                yadd = 260 - 42 * 3
                self.screen.blit(amount_bgrd, (coord.x + 450 - xadd - amount_bgrd.get_width(), coord.y + 400 - yadd - amount_bgrd.get_height()))
        # show cursor
        self.screen.blit(self.cursor, (self.cursor_pos[0], self.cursor_pos[1]))
        #info
        if(not self.battle_engine.last_stack):
            return
        root = pygame.Surface((300, 200))
        #root.fill((int(back_color[0]), int(back_color[1]), int(back_color[2])))
        root.set_colorkey((0,0,0))
        last_stack_coord = CClickableHex.getXYUnitAnim(BHex(self.battle_engine.last_stack.x, self.battle_engine.last_stack.y), self.battle_engine.last_stack)
        start_height = 0
        for text in logger.log_text:
            ftext = self.font.render(text, True, (255, 255, 255))
            root.blit(ftext, (0,start_height))
            start_height = start_height + ftext.get_height() + 5
        xadd = 220 - (44 if self.battle_engine.last_stack.side else -22)
        yadd = 260 - 42 * 3
        self.screen.blit(root, (last_stack_coord.x + 450 - xadd, last_stack_coord.y + 400 - yadd))
         #CClickableHex(self.current_hex.hex_i,self.current_hex.hex_j).getHexXY()
    def update(self):
        self.showBackground()
        self.showBattlefieldObjects()
        # self.showProjectiles(to)
        # updateBattleAnimations()
        # showInterface(to)

    def handleEvents(self):
        # 处理游戏退出
        # 从消息队列中循环取
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.MOUSEMOTION:
                mouse_x, mouse_y = event.pos
                self.handleMouseMotion(mouse_x, mouse_y)
            elif event.type == pygame.MOUSEBUTTONUP:
                cur_stack = self.battle_engine.cur_stack
                if cur_stack.by_AI == 0:
                    act = self.act
                    self.cursor = self.cursor_move[3]
                    self.act = None
                    return act
                else:
                    return self.next_act
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_d:
                    self.battle_engine.dump_battle(self.dump_dir)
                    return
                cur_stack = self.battle_engine.cur_stack
                if cur_stack.by_AI == 0:
                    if event.key == pygame.K_SPACE:
                        return BAction(action_type.defend)
                    if event.key == pygame.K_w:
                        if cur_stack.had_waited:
                            logger.error("can't wait any more!",True)
                        else:
                            return BAction(action_type.wait)

    def shift_attack_pointer(self,sector,mouse_x, mouse_y):
        x = mouse_x - 16
        y = mouse_y - 16
        if sector == 0:
            x -= 6
            y += 16
        if sector == 1:
            x -= 6
            y -= 6
        if sector == 2:
            x += 16
            y -= 6
        if sector == 3:
            x += 16
            y += 11
        if sector == 4:
            x += 16
            y += 16
        if sector == 5:
            x -= 6
            y += 16
        self.cursor_pos = (x,y)
    def handleMouseMotion(self, mouse_x, mouse_y):
        #hover on hex
        if mouse_x < 58 or mouse_y <86 or mouse_x >= 750 or mouse_y >= 545:
            return

        i,j = CClickableHex.XYtoIJ(mouse_x,mouse_y)
        self.current_hex.hex_i = i
        self.current_hex.hex_j = j
        px = mouse_x - (mouse_x - (14 + (22 if i % 2 == 0 else 0))) % 44
        py = mouse_y - (mouse_y - 86) % 42
        self.current_hex.pixels_x = px
        self.current_hex.pixels_y = py
        self.cursor_pos = (mouse_x - 16 , mouse_y - 16)
        #hover on stack
        self.hoveredStack = None
        for stack in self.battle_engine.stacks:
            if self.current_hex.hex_i == stack.y and self.current_hex.hex_j == stack.x:
                self.hoveredStack = stack
        #cursor and action
        cur_stack = self.battle_engine.cur_stack
        bf = cur_stack.get_global_state()
        h = self.current_hex
        if bf[h.hex_i, h.hex_j] == 201:
            if cur_stack.can_shoot():
                if cur_stack.is_half(BHex(h.hex_j,h.hex_i)):
                    self.cursor = self.cursor_shoot[1]
                else:
                    self.cursor = self.cursor_shoot[0]
                self.act = BAction(action_type.shoot, target=self.hoveredStack)
            else:
                bf[cur_stack.y,cur_stack.x] = cur_stack.speed
                subdividingAngle = 2.0 * np.pi / 6.0 # Divide hex in 6 directions
                hexMidX = self.current_hex.pixels_x + 22
                hexMidY = self.current_hex.pixels_y + 21
                cursorHexAngle = np.pi - math.atan2(hexMidY - mouse_y, mouse_x - hexMidX) + subdividingAngle / 2
                sector = int(cursorHexAngle / subdividingAngle) % 6
                # zigzagCorrection =0 if (self.current_hex.hex_i % 2) else 1
                from_dest = self.battle_engine.direction_to_hex(BHex(self.current_hex.hex_j,self.current_hex.hex_i),sector)

                if 0 <= bf[from_dest.y,from_dest.x] < 50:
                    self.cursor = self.cursor_attack[sector]
                    self.shift_attack_pointer(sector,mouse_x,mouse_y)
                    self.act = BAction(action_type.attack, target=self.hoveredStack, dest=from_dest)
                else:
                    self.cursor = self.cursor_move[3]
                    self.act = None
        elif bf[h.hex_i, h.hex_j] == 200 or bf[h.hex_i, h.hex_j] == 400 or bf[h.hex_i, h.hex_j] == 401:
            self.cursor = self.cursor_move[4]
            self.act = None
        elif bf[h.hex_i, h.hex_j] == 800 or bf[h.hex_i, h.hex_j] == 100:
            self.cursor = self.cursor_move[3]
            self.act = None
        elif 0 <= bf[h.hex_i, h.hex_j] < 50:
            if cur_stack.is_fly:
                self.cursor = self.cursor_move[2]
                self.act = BAction(action_type.move, dest=BHex(h.hex_j, h.hex_i))
            else:
                self.cursor = self.cursor_move[1]
                self.act = BAction(action_type.move, dest=BHex(h.hex_j, h.hex_i))
        elif bf[h.hex_i, h.hex_j] < 0:
            self.cursor = self.cursor_move[3]
            self.act = None
        # print("hovered on pixels{},{} location{},{} repixels{},{}".format(mouse_x,mouse_y,i,j,self.current_hex.pixels_x,self.current_hex.pixels_y))

    def handleBattle(self,act,print_act=True):
        if not act:
            return
        battle = self.battle_engine
        battle.doAction(act)
        battle.checkNewRound()
        if battle.check_battle_end():
            return
        self.next_act = battle.cur_stack.active_stack(print_act = print_act)

    def getObstaclePosition(self,img, obinfo):
        if obinfo.isabs:
            return obinfo.width,obinfo.height
        offset = img.get_height() % 42
        if offset > 37:
            offset -=42
        bh = CClickableHex(obinfo.y,obinfo.x)
        x,y = bh.getHexXY()
        y += 42 - img.get_height() + offset
        return x,y
def start_game():
    # 初始化游戏
    pygame.init()  # 初始化pygame
    pygame.display.set_caption('This is my first pyVCMI')  # 设置窗口标题
    battle = Battle(by_AI = [0,1])
    battle.load_battle("ENV/battles/0.json", shuffle_postion=False,load_ai_side=False)
    battle.checkNewRound()
    bi = BattleInterface(battle)
    bi.next_act = battle.cur_stack.active_stack()
    # 事件循环(main loop)
    while bi.running:
        act = bi.handleEvents()
        bi.handleBattle(act)
        if battle.check_battle_end():
            logger.debug("battle end~")
            bi.running = False
            pygame.quit()
        bi.renderFrame()
def start_game_record():
    # 初始化游戏
    pygame.init()  # 初始化pygame
    pygame.display.set_caption('This is my first pyVCMI')  # 设置窗口标题
    battle = Battle(by_AI = [0,1])
    battle.load_battle("ENV/battles/0.json", shuffle_postion=False,load_ai_side=False)
    battle.checkNewRound()
    bi = BattleInterface(battle)
    bi.next_act = battle.cur_stack.active_stack()
    # 事件循环(main loop)
    while bi.running:
        act = bi.handleEvents()
        if act and battle.cur_stack.by_AI == 0:
            masks = battle.get_act_masks(act)
            ind, attri_stack, planes_stack, plane_glb = battle.current_state_feature()
        bi.handleBattle(act)
        if battle.check_battle_end():
            logger.debug("battle end~")
            bi.running = False
            pygame.quit()
        bi.renderFrame()
if __name__ == '__main__':
    start_game()

