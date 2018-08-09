import tensorflow as tf
import numpy as np
a = np.arange(1*36*1).reshape(1,36,1)
#a = np.arange(3*2*4).reshape(3,2,4)
#a = tf.convert_to_tensor(a)
# axis=0
sum1 = tf.cumsum(a,axis=0)
#sum1 = tf.reduce_sum(a,axis=0)
# axis=1
sum2 = tf.cumsum(a, axis=1)
#sum2 = tf.reduce_sum(a, axis=1)
# exclusive=False
sum3 = tf.cumsum(a, exclusive=False)

# exclusive=True
sum4 = tf.cumsum(a, exclusive=True)

# reverse=True
sum5 = tf.cumsum(a, reverse=True)

# exclusive=True, reverse=True
sum6 = tf.cumsum(a, exclusive=True, reverse=True)
init = tf.global_variables_initializer()
with tf.Session() as session:
    session.run(init)
    b = session.run(sum1)
    bb = session.run(sum2)
    pass
