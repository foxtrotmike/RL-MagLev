# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 15:22:11 2018
A simple example of a magnetic levitation system controlled using Reinforcement Learning
@author: Fayyaz Minhas
"""

import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
from maglevEnv import MagLevEnv
import numpy as np

# hyper parameters
EPISODES = 70  # number of episodes
EPS_START = 0.9  # e-greedy threshold start value
EPS_END = 0.01 # e-greedy threshold end value
EPS_DECAY = 1000  # e-greedy threshold decay
GAMMA = 0.98  # Q-learning discount factor
LR = 0.005  # NN optimizer learning rate
BATCH_SIZE = 64  # Q-learning batch size

# if gpu is to be used
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        self.memory.append(transition)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class Network(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(2, 200)
        self.l3 = nn.Linear(200, 2)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = (self.l3(x))
        return x

env = MagLevEnv()
env.referencepoint = 2.0

model = Network()
if use_cuda:
    model.cuda()
memory = ReplayMemory(10000)
optimizer = optim.Adam(model.parameters(), LR)
steps_done = 0
episode_durations = []


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        return model(Variable(state, volatile=True).type(FloatTensor)).data.max(1)[1].view(1, 1)
    else:
        return LongTensor([[random.randrange(2)]])


def run_episode(e, environment):
    ref = environment.referencepoint
    state = environment.reset()    
    steps = 0
    while True:
        steps += 1
        
        action = select_action(FloatTensor([state]))
        a = action.data.numpy()[0,0]
        next_state, reward, done, _ = environment.step(a)



        memory.push((FloatTensor([state]),
                     action,  # action is already a tensor
                     FloatTensor([next_state]),
                     FloatTensor([reward])))

        
        learn()

        state = next_state

        if steps > 500:
            print("Episode %s Final Position Error %s " %(e, np.abs(next_state[1]-ref)))
            episode_durations.append(np.abs(next_state[1]-ref))
            plotError()
            
            break


def learn():
    global GAMMA
    if len(memory) < BATCH_SIZE:
        return
    
    # random transition batch is taken from experience replay memory
    transitions = memory.sample(BATCH_SIZE)
    batch_state, batch_action, batch_next_state, batch_reward = zip(*transitions)

    batch_state = Variable(torch.cat(batch_state))
    batch_action = Variable(torch.cat(batch_action))
    batch_reward = Variable(torch.cat(batch_reward))
    batch_next_state = Variable(torch.cat(batch_next_state))

    # current Q values are estimated by NN for all actions
    current_q_values = model(batch_state).gather(1, batch_action)
    # expected Q values are estimated from actions which gives maximum Q value
    max_next_q_values = model(batch_next_state).detach().max(1)[0]
        

    expected_q_values = batch_reward + (GAMMA * max_next_q_values)

    # loss is measured from error between current and newly expected Q values
    loss = F.smooth_l1_loss(current_q_values, expected_q_values.view(BATCH_SIZE,1))

    # backpropagation of loss to NN
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()



def plotError():
    plt.figure(2)
    plt.clf()
    durations_t = torch.FloatTensor(episode_durations)
    plt.title(str(steps_done))
    plt.xlabel('Episode')
    plt.ylabel('Final Position Error')
    plt.plot(durations_t.numpy())
    # take 100 episode averages and plot them too
    H = 10
    if len(durations_t) >= H:
        means = durations_t.unfold(0, H, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(H-1), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated



for e in range(EPISODES):
    run_episode(e, env)

print('Complete')
#env.render(close=True)
#
plt.pause(0.5)

#%% TESTING 

state = env.reset()
env.position = 0
env.velocity = -2.0
env.mass = 1.0
state = [env.velocity,env.position]
S = [state] #States history for test
for i in range(500):    
    action = select_action(FloatTensor([state]))
    a = action.data.numpy()[0,0]
    state,reward,done,_ = env.step(a)
    S.append(state)
    print(i,state,a,reward,done)
    if done:
        print("out of bounds")
    env.render()
S = np.array(S)    

#%% Plotting the policy in the state space.
x = np.linspace(0, 10, 50)
v = np.linspace(-20, 20, 60)
A = np.zeros((len(x),len(v)))
for i,xi in enumerate(x):
    for j,vj in enumerate(v):
        A[i,j] = select_action(FloatTensor([[vj,xi]])).data.numpy()[0,0]
plt.figure(3)
plt.contourf(v,x,A,levels=[0.1,1]);plt.scatter(S[:,0],S[:,1],c='r'); plt.plot(S[0,0],S[0,1],c = 'k', marker ='*'); plt.scatter(S[-1,0],S[-1,1],c = 'k', marker ='s')
plt.figure(4)
plt.plot(S[:,1])