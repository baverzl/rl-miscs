import os
import argparse

# basic
import math
import random
from itertools import count

# numpy
import numpy as np

# torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision.transforms as T

# env
from obstacle_tower_env import ObstacleTowerEnv

# visualize
import matplotlib.pyplot as plt


# model
from model import DQN
from model import ReplayMemory

env = ObstacleTowerEnv('./obstacle-tower-challenge/ObstacleTower/obstacletower', retro=False, realtime_mode=False)

def set_arguments():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--gamma', type=float, default=0.999)
    parser.add_argument('--eps_start', type=float, default=0.9)
    parser.add_argument('--eps_end', type=float, default=0.05)
    parser.add_argument('--eps_decay', type=int, default=200)
    parser.add_argument('--target_update', type=int, default=10)
   
    parser.add_argument('--memory_size', type=int, default=10000)

    return parser.parse_args()

def get_screen(obs):
    screen = obs[0]

    # Need to preprocess?
    screen = torch.from_numpy(screen)
    
    # [H, W, 3] --> [1, 3, H, W]
    screen = screen.permute(2, 0, 1)
    screen = torch.unsqueeze(screen, dim=0)

    return screen

class Solver(object):
    def __init__(self, args):
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.eps_start = args.eps_start
        self.eps_end = args.eps_end
        self.eps_decay = args.eps_decay
        self.target_update = args.target_update
        
        N= args.memory_size
        
        obs = env.reset()
        init_screen = get_screen(obs)
        _, _, h, w = init_screen.size()

        self.build_model(w, h, N)

    def build_model(self, w, h, N):
        self.policy_net = DQN(h, w)
        self.target_net = DQN(h, w)

        self.opt = optim.RMSprop(policy_net.parameters())
        self.memory = ReplayMemory(N)

        self.steps_done = 0
        self.episode_durations = []

    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - eps_end) * \
                 math.exp(-1. * steps_done / self.eps_decay)
        step_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return policy_net(state).max(1)[1].view(1,1)
        
        else:
            return torch.tensor([[random.randrange(2)]], dtype=torch.long)

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
    
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)))
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size)
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.opt.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.opt.step() 

    def plot_durations():
        plt.figure(2)
        plt.clf()
    
        durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
        plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy())
    
        # Take 100 episode averages and plot them too
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.001)  # pause a bit so that plots are updated
        display.clear_output(wait=True)
        display.display(plt.gcf())

    def exploration(self, num_episodes = 50):
        for i_episode in range(num_episodes):
            env.reset()
            last_screen = get_screen()
            current_screen = get_screen()

            O = current_screen - last_screen

            for t in count():
                # selectr and perform an action
                action = self.select_action(O)
                _, reward, done, _ = env.step(action.item())
                reward = torch.tensor([reward])

                # Observe new state
                last_screen = current_screen
                current_screen = get_screen()
                if not done:
                    next_O = current_screen - last_screen
                else:
                    next_O = None

                self.memory.push(O, action, next_O, reward)

                # Move to the next sate
                O = next_O

                # Perform one step of the optimization
                self.optimize_model()
                if done:
                    self.episode_durations.append(t+1)
                    self.plot_durations()
                    break
                
            if i_episode % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

        print('Complete!')
        env.render()
        env.close()
        plt.ioff()
        plt.show()

if __name__ == '__main__':
    args = set_arguments()
    agent = Solver(args)
    agent.exploration()
