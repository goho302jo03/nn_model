import numpy as np
import time
import math

# prints formatted price
def formatPrice(n):
    # return ("-$" if n < 0 else "$") + "{0:.2f}".format(abs(n))
    return f'{"-$" if n < 0 else "$"} {abs(n):.2f}'

# returns the vector containing stock data from a fixed file
# def getStockDataVec(key):
#     vec = []
#     lines = open(f'data/{key}.csv', 'r').read().splitlines()
#
#     for line in lines[1:]:
#         vec.append(float(line.split(',')[4]))
#
#     return vec
def getStockDataVec(key):
    with open(f'data/{key}.csv', 'r') as f:
        data = [float(v.rstrip('\n').split(',')[2]) for v in f.readlines()[1:]]
    return data

# returns the sigmoid
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# returns an an n-day state representation ending at time t
def getState(data, t, n):
    d = t - n + 1
    block = data[d:t + 1] if d >= 0 else -d * [data[0]] + data[0:t + 1] # pad with t0
    res = []
    for i in range(n - 1):
        res.append(sigmoid(block[i + 1] - block[i]))

    return res
