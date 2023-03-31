# VCNN
英雄无敌3的战场AI,目标是能够代替玩家完成重复单调的野怪战场部分(野怪AI采用行为树脚本),用尽量少的损失拿下野怪(能模仿玩家水平更好).为多人对战减少90%等待时间<br>
环境部分<br>
di-engine 0.2.3
torch 1.8.0
tianshou 0.4.5
        --python部分 /ENV <br>
        --c++部分 /VCCC 精力有限,只重写了计算量最大的部分<br>
AI部分<br>
        PG_Model/h3_ppo.py<br>
食用方法<br>
在VCNN目录下 python PG_Model/h3_ppo.py 入口函数start_train() <br>
use_expert_data=True go_explore模式模仿学习 / use_expert_data=False 纯RL<br>

