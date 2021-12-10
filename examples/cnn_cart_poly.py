from os import environ, stat
import gym
import copy
from torch.nn.modules.activation import ReLU
from torch.nn.modules.flatten import Flatten
import tqdm
import time
import torch
import random
import operator
import numpy as np
import torch.nn as nn
import torch.optim as optim
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


class AgentNet(nn.Module):
    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.base = self._create_conv_base()
        self.head = self._create_head()

    def forward(self, screen: torch.Tensor, action: torch.Tensor):
        screen /= 255.0
        dense_out = self.base(screen.to(self.device))
        dense_out = torch.cat((dense_out, action.to(self.device)), dim=1)
        output = self.head(dense_out)
        return output

    @staticmethod
    def _create_conv_base():
        return nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
        )

    @staticmethod
    def _create_head():
        return nn.Sequential(
            nn.Linear(108289, 4), nn.ReLU(), nn.Linear(4, 1), nn.Sigmoid()
        )


class CartPoleAgent:
    def __init__(self, epsilon: float = 0.1, gamma: float = 0.9):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.prediction_net = AgentNet(self.device).to(self.device)
        self.target_net = AgentNet(self.device).to(self.device)
        self.epsilon = epsilon
        self.gamma = gamma

    def calculate_prediction_q_value(self, screen: np.ndarray, action: int):
        tensor = self._convert_to_tensor(screen)
        action = torch.FloatTensor([action]).unsqueeze(0)
        return self.prediction_net(tensor, action)

    def calculate_target_q_value(self, screen: np.ndarray, action: int):
        tensor = self._convert_to_tensor(screen)
        action = torch.FloatTensor([action]).unsqueeze(0)
        return self.prediction_net(tensor, action)

    def get_max_value_action(self, screen: np.ndarray) -> torch.Tensor:
        q_values = []
        for action in range(2):
            q_values.append(self.calculate_target_q_value(screen, action))
        _, max_value = max(enumerate(q_values), key=operator.itemgetter(1))
        return max_value

    def get_action_max_value(self, screen: np.ndarray) -> int:
        q_values = []
        for action in range(2):
            q_values.append(self.calculate_prediction_q_value(screen, action))
        action, _ = max(enumerate(q_values), key=operator.itemgetter(1))
        return action

    def epsilon_greedy(self, screen: np.ndarray, step_number: int = 1) -> int:
        if random.random() < self.epsilon / step_number:
            return round(random.random())
        else:
            action = self.get_action_max_value(screen)
            return action

    def save_weights(self):
        torch.save(self.prediction_net.state_dict(), "../weights/cart_pole_cnn.pth")

    def _convert_to_tensor(self, screen: np.ndarray):
        return (
            torch.from_numpy(screen.astype(np.float64))
            .transpose(1, 2)
            .transpose(0, 1)
            .float()
            .unsqueeze(dim=0)
        )

    @contextmanager
    def enable_train_mode(self):
        try:
            self.prediction_net = self.prediction_net.train(True)
            yield
        finally:
            self.prediction_net = self.prediction_net.train(False)


def train(env: gym.Env, agent: CartPoleAgent, num_episodes=20000):
    optimizer = optim.Adam(
        agent.prediction_net.parameters(), lr=0.001, weight_decay=0.001
    )
    criterion = nn.MSELoss()
    optimizer.zero_grad()
    with agent.enable_train_mode():
        rewards = [0]
        iters = [0]
        losses = [0]
        for iter_num in tqdm.tqdm(
            range(1, num_episodes + 1), desc="Train Loss: {}".format(losses[-1])
        ):
            if iter_num % 10 == 0:
                optimizer.step()
                optimizer.zero_grad()
                agent.target_net.load_state_dict(agent.prediction_net.state_dict())
            done = False
            print(env.reset())
            screen = env.render(mode="rgb_array")
            total_reward = 0
            while not done:
                # environment interaction
                action = agent.epsilon_greedy(screen)
                old_screen = screen.copy()
                _, reward, done, _ = env.step(action)
                screen = env.render(mode="rgb_array")
                with torch.no_grad():
                    best_next_value = agent.get_max_value_action(screen)

                # if type(env.steps_beyond_done) == int:
                #     done = env.steps_beyond_done > 100

                # creating the target
                if done:
                    y_target = reward
                else:
                    y_target = reward + agent.gamma * best_next_value

                y_target = torch.FloatTensor([y_target]).to(agent.device)
                prediction = agent.calculate_prediction_q_value(old_screen, action)
                total_reward += reward
                loss = criterion(prediction, y_target)
                print(
                    "Target -> {:2.10} \tPrediction -> {:2.10} \tLoss -> {:2.10}".format(
                        y_target.item(), prediction.item(), loss.item()
                    )
                )
                loss.backward()

            iters.append(iter_num)
            rewards.append(total_reward)
            losses.append(loss.detach().item())
    plt.plot(iters, losses)
    plt.show()

    plt.clf()
    plt.plot(iters, rewards)
    plt.show()
    env.close()


def demonstrate(env: gym.Env, agent: CartPoleAgent):
    try:
        while True:
            done = False
            np.random.seed(random.randint(0, 10000))
            state = env.reset()
            while not done:
                env.render()
                screen = env.render("rgb_array")
                action = agent.get_action_max_value(screen)
                state, reward, _, info = env.step(action)
                if type(env.steps_beyond_done) == int:
                    done = env.steps_beyond_done > 100
                time.sleep(1 / 60)
    except KeyboardInterrupt:
        env.close()


def main():
    environment = gym.make("CartPole-v0")
    agent = CartPoleAgent()
    np.random.random()
    demonstrate(environment, agent)
    train(environment, agent)
    agent.save_weights()
    # agent.prediction_net.load_state_dict(torch.load("../weights/cart_pole_ann.pth"))
    # agent.target_net.load_state_dict(torch.load("../weights/cart_pole_ann.pth"))
    demonstrate(environment, agent)


if __name__ == "__main__":
    main()
