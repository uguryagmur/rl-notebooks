from os import environ, stat
import gym
from torch.nn.modules.activation import LeakyReLU
import tqdm
import time
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.functional as func
import matplotlib.pyplot as plt

from contextlib import contextmanager

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


class CartPoleAgent:
    state = None
    q_values = None
    next_q_values = None
    action = None

    def __init__(self, epsilon: float = 0.1, gamma: float = 0.9):
        self.device = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.policy_net = self.create_neural_net().to(self.device)
        self.target_net = self.create_neural_net().eval().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.epsilon = epsilon
        self.gamma = gamma

    def create_neural_net(self) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(108288, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 2),
        )

    def get_policy_result(self, state: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(state.astype(np.float64))
        tensor /= 255.0
        tensor = tensor.transpose(2, 1).transpose(0, 1).unsqueeze(dim=0).float()
        return self.policy_net(tensor.to(self.device))

    def get_target_result(self, state: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(state.astype(np.float64))
        tensor /= 255.0
        tensor = tensor.transpose(2, 1).transpose(0, 1).unsqueeze(dim=0).float()
        return self.target_net(tensor.to(self.device))

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

    @contextmanager
    def enable_train_mode(self):
        try:
            self.policy_net = self.policy_net.train(True)
            yield
        finally:
            self.policy_net = self.policy_net.train(False)


def train(env: gym.Env, agent: CartPoleAgent, num_episodes=10000):
    optimizer = optim.Adam(agent.policy_net.parameters(), lr=0.001)
    criterion = nn.SmoothL1Loss()

    try:
        with agent.enable_train_mode():
            losses = [0]
            avg_loss = 0.0
            avg_iter = 0
            bar = tqdm.tqdm(range(1, num_episodes + 1))
            for iter_num in bar:
                if iter_num % 10 == 0:
                    agent.target_net.load_state_dict(agent.policy_net.state_dict())
                done = False
                _ = env.reset()
                state = env.render(mode="rgb_array")
                tensor = state.copy()
                total_reward = 0
                while not done:
                    old_state = state.copy()
                    action = agent.epsilon_greedy(tensor)
                    _, reward, done, _ = env.step(action)
                    state = env.render(mode="rgb_array")
                    tensor = state - old_state
                    best_next_value = agent.get_best_value_from_target(state).item()

                    if done:
                        y_target = -1
                    else:
                        y_target = reward + agent.gamma * best_next_value

                    y_target = torch.FloatTensor([y_target]).to(agent.device)
                    prediction = torch.max(agent.get_policy_result(old_state))
                    total_reward += reward
                    loss = criterion(prediction, y_target)
                    loss.backward()
                    avg_loss += loss.item()
                    avg_iter += 1
                    losses.append(loss.detach().item())
                    optimizer.step()
                    optimizer.zero_grad()
                bar.set_description("Loss -> {:2.10f}".format(avg_loss / avg_iter))
        plt.plot(losses)
        plt.show()
        env.close()
    except KeyboardInterrupt:
        optimizer.zero_grad()
        plt.plot(losses)
        plt.show()
        env.close()


def demonstrate(env: gym.Env, agent: CartPoleAgent):
    try:
        while True:
            done = False
            np.random.seed(random.randint(0, 10000))
            _ = env.reset()
            state = env.render("rgb_array")
            while not done:
                env.render()
                action = torch.argmax(agent.get_policy_result(state)).item()
                _, reward, _, info = env.step(action)
                state = env.render("rgb_array")
                if type(env.steps_beyond_done) == int:
                    done = env.steps_beyond_done > 100
                time.sleep(1 / 60)
    except KeyboardInterrupt:
        env.close()


def main():
    environment = gym.make("CartPole-v0")
    agent = CartPoleAgent()
    demonstrate(environment, agent)
    train(environment, agent)
    agent.save_weights()
    # agent.policy_net.load_state_dict(torch.load("../weights/cart_pole_ann.pth"))
    # agent.target_net.load_state_dict(torch.load("../weights/cart_pole_ann.pth"))
    demonstrate(environment, agent)
    breakpoint()


if __name__ == "__main__":
    main()
