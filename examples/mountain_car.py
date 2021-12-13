from os import environ, stat
import gym
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
        Num    Observation               Min            Max
        0      Car Position              -1.2           0.6
        1      Car Velocity              -0.07          0.07

Environment actions:
    Type: Discrete(2)
        Num    Action
        0      Accelerate to the Left
        1      Don't accelerate
        2      Accelerate to the Right
        
        Note: This does not affect the amount of velocity affected by the
        gravitational pull acting on the car.

Environment reward:
        Reward of 0 is awarded if the agent reached the flag (position = 0.5) on top of the mountain.
        Reward of -1 is awarded if the position of the agent is less than 0.5.

Environment starting state:
    The position of the car is assigned a uniform random value in [-0.6 , -0.4].
    The starting velocity of the car is always assigned to 0

Episode Termination:
    The car position is more than 0.5
    Episode length is greater than 200
"""


class MountainCarAgent:
    state = None
    q_values = None
    next_q_values = None
    action = None

    def __init__(self, epsilon: float = 0.1, gamma: float = 0.9):
        self.policy_net = self.create_neural_net()
        self.target_net = self.create_neural_net().eval()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.epsilon = epsilon
        self.gamma = gamma

    def create_neural_net(self) -> nn.Module:
        return nn.Sequential(
            nn.Linear(2, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 3),
        )

    def get_policy_result(self, state: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(state)
        tensor[..., 0] = (tensor[..., 0] + 1.2) / 1.8
        tensor[..., 0] = (tensor[..., 0] + 0.07) / 0.14
        return self.policy_net(tensor)

    def get_target_result(self, state: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(state)
        tensor[..., 0] = (tensor[..., 0] + 1.2) / 1.8
        tensor[..., 0] = (tensor[..., 0] + 0.07) / 0.14
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

    @contextmanager
    def enable_train_mode(self):
        try:
            self.policy_net = self.policy_net.train(True)
            yield
        finally:
            self.policy_net = self.policy_net.train(False)


def train(env: gym.Env, agent: MountainCarAgent, num_episodes=8000):
    optimizer = optim.Adam(agent.policy_net.parameters(), lr=0.001)
    criterion = nn.SmoothL1Loss()

    try:
        with agent.enable_train_mode():
            losses = [0]
            bar = tqdm.tqdm(range(1, num_episodes + 1))
            for iter_num in bar:
                if iter_num % 10 == 0:
                    agent.target_net.load_state_dict(agent.policy_net.state_dict())
                done = False
                state = env.reset()
                total_reward = 0
                avg_loss = 0.0
                avg_iter = 0
                while not done:
                    old_state = state.copy()
                    action = agent.epsilon_greedy(old_state)
                    state, reward, done, _ = env.step(action)
                    best_next_value = agent.get_best_value_from_target(state).item()

                    if done:
                        y_target = state[0]
                    else:
                        y_target = state[0] + agent.gamma * best_next_value

                    y_target = torch.FloatTensor([y_target])
                    prediction = torch.max(agent.get_policy_result(old_state))
                    total_reward += reward
                    loss = criterion(prediction, y_target)
                    env.render()
                    loss.backward()
                    avg_loss += loss.item()
                    avg_iter += 1
                    optimizer.step()
                    optimizer.zero_grad()
                bar.set_description("Loss -> {:2.10f}".format(avg_loss / avg_iter))
                losses.append(avg_loss / avg_iter)
        plt.plot(losses)
        plt.show()
        env.close()
    except KeyboardInterrupt:
        optimizer.zero_grad()
        plt.plot(losses)
        plt.show()
        env.close()


def demonstrate(env: gym.Env, agent: MountainCarAgent):
    try:
        while True:
            done = False
            np.random.seed(random.randint(0, 10000))
            state = env.reset()
            print(state)
            while not done:
                env.render()
                action = torch.argmax(agent.get_policy_result(state)).item()
                state, reward, _, info = env.step(action)
                time.sleep(1 / 60)
    except KeyboardInterrupt:
        env.close()


def main():
    environment = gym.make("MountainCar-v0")
    agent = MountainCarAgent()
    # demonstrate(environment, agent)
    train(environment, agent)
    agent.save_weights()
    # agent.policy_net.load_state_dict(torch.load("../weights/cart_pole_ann_rbf.pth"))
    # agent.target_net.load_state_dict(torch.load("../weights/cart_pole_ann.pth"))
    demonstrate(environment, agent)


if __name__ == "__main__":
    main()
