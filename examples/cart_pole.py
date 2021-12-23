import gym
import tqdm
import time
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from typing import Any
from contextlib import contextmanager
from torch.utils.tensorboard import SummaryWriter

"""
Cart Pole System Notes

Environment observation:
    Type: Box(4)
    Num     Observation             Min         Max
    0       Cart Position           -4.8        4.8
    1       Cart Velocity           -Inf        Inf
    2       Cart Angle              -0.418 rad  0.418 rad
    3       Pole Anguler Velocity   -Inf        Inf

Environment actions:
    Type: Discrete(2)
    Num     Action
    0       Push cart to the left
    1       Push cart to the right

Environment reward:
    Reward is 1 for every step taken, including the termination step

Environment starting state:
    All observations are assigned a uniform random value in [-0.05..0.05]

Episode Termination:
    Pole Angle is more than 12 degrees.
    Cart Position is more than 2.4 (center of the cart reaches the edge of
    the display).
    Episode length is greater than 200.
    Solved Requirements:
    Considered solved when the average return is greater than or equal to
    195.0 over 100 consecutive trials.
"""


class ReplayMemory:
    def __init__(self, size: int = 100):
        self.size = 0
        self.memory = torch.zeros((size, 5))

    def append(self, element: Any):
        if self.size < self.memory.size(0):
            self.size += 1
        stacked = torch.concat(element, dim=0)
        self.memory = torch.concat((stacked.unsqueeze(0), self.memory[:-1]), dim=0)

    def sample(self, batch_size: int = 16):
        perm = torch.randperm(self.memory.size(0))
        sample = self.memory[perm[:batch_size]]
        return sample[..., 0], sample[..., 1:]

    def is_full(self) -> bool:
        return self.size == self.memory.size(0)


class CartPoleAgent:
    state = None
    q_values = None
    next_q_values = None
    action = None
    writer = SummaryWriter()

    def __init__(self, epsilon: float = 0.1, gamma: float = 0.9):
        self.policy_net = self.create_neural_net()
        self.target_net = self.create_neural_net().eval()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.epsilon = epsilon
        self.gamma = gamma

    def create_neural_net(self) -> nn.Module:
        return nn.Sequential(
            nn.Linear(4, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 2),
        )

    def normalize_state(self, state):
        tensor = torch.from_numpy(state)
        tensor[..., 0] /= 4.8
        tensor[..., 1] = torch.tanh(tensor[..., 1])
        tensor[..., 2] /= 0.418
        tensor[..., 3] = torch.tanh(tensor[..., 3])
        return tensor

    def get_policy_result(self, state: np.ndarray) -> torch.Tensor:
        tensor = self.normalize_state(state)
        return self.policy_net(tensor)

    def get_target_result(self, state: np.ndarray) -> torch.Tensor:
        tensor = self.normalize_state(state)
        return self.target_net(tensor)

    def get_best_value_from_target(self, state: np.ndarray) -> torch.Tensor:
        q_values = self.get_target_result(state)
        max_value = torch.max(q_values)
        return max_value

    def get_best_action(self, state: np.ndarray) -> torch.Tensor:
        q_values = self.get_policy_result(state)
        best_action = torch.argmax(q_values, dim=-1)
        return best_action

    def epsilon_greedy(self, state: np.ndarray, step_number: int = 1) -> int:
        if random.random() < self.epsilon / step_number:
            return round(random.random())
        else:
            with torch.no_grad():
                action = self.get_best_action(state).item()
            return action

    def save_weights(self):
        torch.save(self.policy_net.state_dict(), "../weights/cart_pole_ann.pth")

    def _normalize(self, state: np.ndarray):
        state[0] = (state[0] + 4.8) / 9.6
        state[1] /= np.finfo(np.float64).max
        state[2] = (state[0] + 0.418) / 0.836
        state[3] /= np.finfo(np.float64).max
        return state

    @contextmanager
    def enable_train_mode(self):
        try:
            self.policy_net = self.policy_net.train(True)
            yield
        finally:
            self.policy_net = self.policy_net.train(False)


def train(env: gym.Env, agent: CartPoleAgent, num_episodes=100000):
    optimizer = optim.RMSprop(agent.policy_net.parameters(), lr=0.005)
    criterion = nn.SmoothL1Loss()

    try:
        with agent.enable_train_mode():
            losses = [0]
            bar = tqdm.tqdm(range(1, num_episodes + 1))
            memory = ReplayMemory(300)
            for iter_num in bar:
                if iter_num % 10 == 0:
                    agent.target_net.load_state_dict(agent.policy_net.state_dict())
                done = False
                state = env.reset()
                total_reward = 0
                best_reward = 0
                while not done:
                    old_state = state.copy()
                    action = agent.epsilon_greedy(old_state)
                    state, reward, done, _ = env.step(action)
                    best_next_value = agent.get_best_value_from_target(state).item()

                    if done:
                        y_target = -1
                    else:
                        y_target = reward + agent.gamma * best_next_value

                    y_target = torch.FloatTensor([y_target])
                    env.render()
                    total_reward += reward
                    memory.append([y_target, torch.from_numpy(old_state)])
                if memory.is_full():
                    targets, states = memory.sample()
                    prediction, _ = torch.max(
                        agent.get_policy_result(states.numpy()), dim=-1
                    )
                    loss = criterion(prediction, targets)
                    loss.backward()
                    losses.append(loss.detach().item())
                    optimizer.step()
                    optimizer.zero_grad()
                    agent.writer.add_scalar("loss", loss.item(), iter_num)
                    agent.writer.add_scalar(
                        "total_reward", total_reward / 200, iter_num
                    )
                    bar.set_description(
                        "Loss -> {:1.6f} Reward -> {:1.3f}".format(
                            loss.item(), total_reward / 200
                        )
                    )
                    if total_reward >= best_reward:
                        agent.save_weights()
                        best_reward = total_reward
        plt.plot(losses)
        plt.show()
    except KeyboardInterrupt:
        plt.plot(losses)
        plt.show()


def demonstrate(env: gym.Env, agent: CartPoleAgent):
    try:
        while True:
            done = False
            np.random.seed(random.randint(0, 10000))
            state = env.reset()
            print(state)
            while not done:
                env.render()
                action = torch.argmax(agent.get_policy_result(state)).item()
                state, reward, _done, info = env.step(action)
                if type(env.steps_beyond_done) == int:
                    print(reward)
                    done = env.steps_beyond_done > 100
                time.sleep(1 / 60)
    except KeyboardInterrupt:
        env.close()


def main():
    environment = gym.make("CartPole-v0")
    agent = CartPoleAgent()
    # agent.policy_net.load_state_dict(torch.load("../weights/ann_cart_pole_final.pth"))
    train(environment, agent)
    torch.save(agent.policy_net.state_dict(), "../weights/ann_cart_pole_final.pth")
    demonstrate(environment, agent)


if __name__ == "__main__":
    main()
