import platform
Linux = "Linux" == platform.system()
import sys
if Linux:
    sys.path.extend(['/home/enigma/work/project/VCNN/','/home/enigma/work/project/VCNN/VCCC/VCbattle/build'])
else:
    sys.path.extend(['D:\\project\\VCNN', 'D:\\project\\VCNN\\VCCC\\x64\\Release'])
import VCbattle #as numpy_demo2
import numpy as np


# var1 = numpy_demo2.add_arrays_1d(np.array([1, 3, 5, 7, 9]),
#                                  np.array([2, 4, 6, 8, 10]))
# print('-'*50)
# print('var1', var1)
#
# var2 = numpy_demo2.add_arrays_2d(np.array(range(0,16)).reshape([4, 4]),
#                                  np.array(range(20,36)).reshape([4, 4]))
# print('-'*50)
# print('var2', var2)
#
# input1 = np.array(range(0, 48)).reshape([4, 4, 3])
# input2 = np.array(range(50, 50+48)).reshape([4, 4, 3])
# var3 = numpy_demo2.add_arrays_3d(input1,
#                                  input2)
# print('-'*50)
# print('var3', var3)
from H3_battle import BStack
al = []
a = BStack()
a.side = 0
a.amount = 3
a.x = 1
a.y = 3
al.append(a)
a = BStack()
a.side = 1
a.amount = 5
a.x = 15
a.y = 2
al.append(a)
a = BStack()
a.side = 0
a.amount = 6
a.x = 1
a.y = 5
a.speed = 10
al.append(a)
bf = VCbattle.get_global_state(a,al)
