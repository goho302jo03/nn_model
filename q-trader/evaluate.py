import keras
from keras.models import load_model

from agent.agent import Agent
from functions import *
import sys

if len(sys.argv) != 4:
    print('Usage: python evaluate.py [stock] [model] [window_size]')
    exit()

stock_name, model_name, window_size = sys.argv[1], sys.argv[2], int(sys.argv[3])

agent = Agent(window_size, 10, True, model_name)
data = getStockDataVec(stock_name)[-600000:]
l = len(data) - 1

state = getState(data, 0, window_size + 1)
total_profit = 0
agent.inventory = []

for t in range(l):
    action = agent.choose_action(state)

    # sit
    next_state = getState(data, t + 1, window_size + 1)
    profit = 0
    reward = 0

    if 1 == action: # buy
        agent.inventory.append(data[t])
        print(f'Buy: {formatPrice(data[t])}')

    elif 2 == action and len(agent.inventory): # sell
        for bought_price in agent.inventory:
            profit += data[t] - bought_price
        agent.inventory = []
        reward = max(profit, 0)
        total_profit += profit
        print(f'Sell: {formatPrice(data[t])} | Profit: {formatPrice(profit)}')

    done = True if t == l - 1 else False
    agent.store_transition(state, action, reward, next_state)
    state = next_state

    if done:
        print('-'*50)
        print(f'Total Profit: {formatPrice(total_profit)}')
        print('-'*50)
