from Environment import *
import os
import torch
from torch import nn
from torch import optim as optim
from torchvision import transforms
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt

ROOT_DIR = os.path.dirname(__file__)


class DQN(nn.Module):

    def __init__(self, lr, input_dims, n_actions):
        super(DQN, self).__init__()
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.conv1 = nn.Conv2d(input_dims[0], 32, (8, 8), stride = (4, 4))
        self.conv2 = nn.Conv2d(32, 64, (4, 4), stride = (2, 2))
        self.conv3 = nn.Conv2d(64, 64, (3, 3), stride = (1, 1))
        fc_input_dims = self.calculate_conv_output_dims(input_dims)
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fc_input_dims, 512),
            nn.ReLU()
        )
        self.V = nn.Linear(512, 1)
        self.A = nn.Linear(512, n_actions)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.optimizer = optim.Adam(self.parameters(), lr = lr)
        self.to(self.device)
        self.loss = nn.MSELoss()

    def calculate_conv_output_dims(self, input_dims):
        state = torch.zeros(1, *input_dims)
        dims = self.conv1(state)
        dims = self.conv2(dims)
        dims = self.conv3(dims)
        return int(np.prod(dims.size()))

    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.relu(self.conv2(X))
        X = F.relu(self.conv3(X))
        X = self.mlp(X)
        V = self.V(X)
        A = self.A(X)
        return V, A


class ReplayBuffer:
    def __init__(self, max_size, input_shape):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = torch.zeros((self.mem_size, *input_shape), dtype = torch.float32)
        self.new_state_memory = torch.zeros((self.mem_size, *input_shape), dtype = torch.float32)

        self.action_memory = torch.zeros(self.mem_size, dtype = torch.int64)
        self.reward_memory = torch.zeros(self.mem_size, dtype = torch.float32)
        self.terminal_memory = torch.zeros(self.mem_size, dtype = torch.bool)
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([84, 84]),
            transforms.Grayscale(num_output_channels = 1)
        ])

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = torch.tensor(action)
        self.reward_memory[index] = torch.tensor(reward)
        self.terminal_memory[index] = torch.tensor(done)
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace = False)
        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal


class Agent:

    def __init__(self, gamma, lr, input_dims, batch_size, n_actions, tau=0.8, replay_mem_size=1000, epsilon=1.0):
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = replay_mem_size
        self.tau = tau
        self.target_update_counter = 0
        self.Q_eval = DQN(lr, input_dims, n_actions)
        self.Q_target = DQN(lr, input_dims, n_actions)
        self.memory = ReplayBuffer(self.mem_size, input_dims)
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([84, 84]),
            transforms.Grayscale(num_output_channels = 1)
        ])

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = observation.unsqueeze(dim = 0).to(self.Q_eval.device)
            _, advantages = self.Q_eval(state.float())
            action = torch.argmax(advantages).item()
        else:
            action = np.random.choice(self.action_space)
        return action

    def update_replay_memory(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def sample_memory(self):
        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)
        states = state.to(self.Q_eval.device)
        rewards = reward.to(self.Q_eval.device)
        dones = done.to(self.Q_eval.device)
        actions = action.to(self.Q_eval.device)
        states_ = new_state.to(self.Q_eval.device)
        return states, actions, rewards, states_, dones

    def learn(self):
        if self.memory.mem_cntr < 10000:
            return
        self.Q_eval.optimizer.zero_grad()
        states, actions, rewards, states_, dones = self.sample_memory()
        indices = np.arange(self.batch_size)
        V_s, A_s = self.Q_eval.forward(states)
        V_s_, A_s_ = self.Q_target.forward(states_)

        q_pred = torch.add(V_s, (A_s - torch.mean(A_s, dim=1, keepdim = True)))[indices, actions]
        q_next = torch.add(V_s_, (A_s_ - torch.mean(A_s_, dim = 1, keepdim = True)))
        q_target = rewards + self.gamma * torch.max(q_next, dim = 1)[0].detach()
        q_target[dones] = 0.0

        loss = self.Q_eval.loss(q_target, q_pred).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()

    def train_targetnet(self):
        for target_param, local_param in zip(self.Q_target.parameters(), self.Q_eval.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)


def decay(value, time_step, thr, mode='min', constant=0.001):
    if mode == 'max':
        constant = -1 * constant
    value = value * np.exp(-constant * time_step)
    if mode == 'max':
        return min(value, thr)
    else:
        return max(value, thr)


env = SnakeEnv()
agent = Agent(gamma = 0.9, lr = 0.0001, input_dims = (4, 84, 84), batch_size = 72, n_actions = 4, tau = 0.8,
              replay_mem_size = 10000, epsilon = 1)
gamma_start = agent.gamma
epsilon_start = agent.epsilon
tau_start = agent.tau
STATS_EVERY = 1
scores, eps_history = [], []
EPISODES = 1000000
for episode in range(EPISODES):
    score = 0
    done = False
    image = env.reset(stop_render = True)
    image = agent.preprocess(image) / 255
    state = torch.cat(tuple(image for _ in range(4)))[:, :, :]
    if episode % 1 == 0:
        env.render()
    while not done:
        action = agent.choose_action(state)
        new_state, reward, done = env.step(action)
        new_state = agent.preprocess(new_state) / 255
        next_state = torch.cat((state[1:, :, :], new_state))[:, :, :]
        agent.update_replay_memory(state, action, reward, next_state, done)
        agent.learn()
        state = next_state
        score += reward
    scores.append(score)
    eps_history.append(agent.epsilon)
    avg_score = np.mean(scores[-STATS_EVERY:])
    if episode % STATS_EVERY == 0:
        print("episode : {} | score : {} | average score :{} | epsilon : {} | gamma : {} | tau : {}".format(
            episode, score, avg_score, agent.epsilon, agent.gamma, agent.tau))
    # decay tau,gamma,epsilon
    if agent.memory.mem_cntr > 10000:
        agent.tau = decay(tau_start, episode + 1, 0.01, constant = 0.00001)
        agent.gamma = decay(gamma_start, episode + 1, 0.999, mode = 'max', constant = 0.00001)
        agent.epsilon = decay(epsilon_start, episode + 1, 0.001, constant = 0.00001)

    if episode % 2 == 0:
        agent.train_targetnet()
        torch.save(agent.Q_eval.state_dict(), os.path.realpath(os.path.join(ROOT_DIR, "FLAPPY-BIRD-DDQL.pt")))

    if episode % 1 == 0:
        env.Stop_render()

torch.save(agent.Q_eval.state_dict(), os.path.realpath(os.path.join(ROOT_DIR, "FLAPPY-BIRD-DDQL.pt")))
x = [i + 1 for i in range(EPISODES)]
plt.plot(x, eps_history, label = "epsilon")
plt.plot(x, scores, label = "Rewards")
plt.legend(loc = 4)
plt.savefig(os.path.realpath(os.path.join(ROOT_DIR, "Statistics.png")))
plt.show()
