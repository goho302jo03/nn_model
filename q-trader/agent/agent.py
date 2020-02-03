import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

class DQN(nn.Module):
    def __init__(self, input_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 8)
        self.fc4 = nn.Linear(8, 3)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)
        return out


class DuelingDQN(nn.Module):
    def __init__(self, input_size):
        super(DuelingDQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 16)
        self.fc2 = nn.Linear(16, 32)
        self.fc3 = nn.Linear(32, 64)
        self.fc4 = nn.Linear(64, 128)
        self.v_fc1 = nn.Linear(128, 64)
        self.v_fc2 = nn.Linear(64, 1)
        self.a_fc1 = nn.Linear(128, 64)
        self.a_fc2 = nn.Linear(64, 3)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.relu(self.fc2(out))
        out = self.relu(self.fc3(out))
        out = self.relu(self.fc4(out))

        v = self.relu(self.v_fc1(out))
        v = self.relu(self.v_fc2(v))

        a = self.relu(self.a_fc1(out))
        a = self.relu(self.a_fc2(a))

        out = v + (a - a.mean(dim=1, keepdim=True))
        return out


class Agent:
    def __init__(self, state_size, memory_capacity=500, is_eval=False, model_name=''):
        self.state_size = state_size # normalized previous days
        self.action_size = 3 # sit, buy, sell
        self.batch_size = 32
        self.learn_step_counter = 0
        self.memory_counter = 0
        self.t_replace_iter = 100
        self.memory_capacity = memory_capacity
        self.memory = np.zeros((memory_capacity, state_size*2+2))
        self.inventory = []
        self.is_eval = is_eval
        self.double = True
        self.dueling = True

        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9999

        if is_eval:
            self.e_model = DQN(state_size)
            self.e_model.load_state_dict(torch.load(f'./models/{model_name}'))
        else:
            if self.dueling:
                self.t_model, self.e_model = DuelingDQN(state_size), DuelingDQN(state_size)
            else:
                self.t_model, self.e_model = DQN(state_size), DQN(state_size)
            self.optimizer = torch.optim.Adam(self.e_model.parameters(), lr=0.01)
            self.loss_func = nn.MSELoss()

    def choose_action(self, state):
        if self.is_eval or np.random.uniform() > self.epsilon:
            state = torch.unsqueeze(torch.FloatTensor(state), 0)
            if self.is_eval:
                actions_value = self.e_model(state)
            else:
                actions_value = self.e_model.forward(state)
            action = torch.max(actions_value, 1)[1].data.numpy()[0]
        else:
            action = random.randrange(self.action_size)
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, s_, [a, r]))
        idx = self.memory_counter % self.memory_capacity
        self.memory[idx, :] = transition
        self.memory_counter += 1

    def learn(self):
        if self.learn_step_counter % self.t_replace_iter == 0:
            self.t_model.load_state_dict(self.e_model.state_dict())
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
        self.learn_step_counter += 1

        sample_idx = np.random.choice(self.memory_capacity, self.batch_size, replace=False)
        b_memory = self.memory[sample_idx, :]
        b_s = torch.FloatTensor(b_memory[:, :self.state_size])
        b_s_ = torch.FloatTensor(b_memory[:, self.state_size:-2])
        b_a = torch.LongTensor(b_memory[:, -2:-1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, -1:])

        q_eval = self.e_model(b_s).gather(1, b_a)
        q_target_next = self.t_model(b_s_).detach()

        if self.double:
            eval_next_action = self.e_model(b_s_).max(1)[1].view(self.batch_size, 1)
            q_target = b_r + self.gamma * q_target_next.gather(1, eval_next_action)
        else:
            q_target = b_r + self.gamma * q_target_next.max(1)[0].view(self.batch_size, 1)

        loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

