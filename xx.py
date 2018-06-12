# import math
# def is_prime(number):
#     if number > 1:
#         if number == 2:
#             return True
#         if number % 2 == 0:
#             return False
#         for current in range(3, int(math.sqrt(number) + 1), 2):
#             if number % current == 0:
#                 return False
#         return True
#     return False
# def get_primes(number):
#     print("iam here")
#     while True:
#         if is_prime(number):
#             print("iam here 2")
#             number = yield number #
#         number += 1
# def print_successive_primes(iterations, base=10):
#     prime_generator = get_primes(base)
#     print("iam here 3")
#     print(prime_generator.send(None))
#     #print(next(prime_generator))
#     for power in range(iterations):
#         print(prime_generator.send(base ** power))
#
# if __name__ == '__main__':
#     print_successive_primes(4,10)

import random


def get_data():
    """返回0到9之间的3个随机数"""
    return random.sample(range(10), 3)


def consume():
    """显示每次传入的整数列表的动态平均值"""
    running_sum = 0
    data_items_seen = 0

    while True:
        data = yield
        data_items_seen += len(data)
        running_sum += sum(data)
        print('The running average is {}'.format(running_sum / float(data_items_seen)))


def produce(consumer):
    """产生序列集合，传递给消费函数（consumer）"""
    while True:
        data = get_data()
        print('Produced {}'.format(data))
        consumer.send(data)
        yield


if __name__ == '__main__':
    consumer = consume()
    # consumer.send(None)
    producer = produce(consumer)

    for _ in range(10):
        print('Producing...')
        next(producer)