from os import environ, stat
import gym
import copy
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


class CartPoleAgent:
    def __init__(self, epsilon: float = 0.1, gamma: float = 0.9):
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.prediction_net = self.create_neural_net().eval()
        self.target_net = copy.deepcopy(self.prediction_net).eval()
        self.epsilon = epsilon
        self.gamma = gamma

    def create_neural_net(self) -> nn.Module:
        return nn.Sequential(
            nn.Linear(6, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def calculate_prediction_q_value(self, state: np.ndarray, action: int):
        tensor = self._create_state_action_vector(state, action)
        return self.prediction_net(tensor)

    def calculate_target_q_value(self, state: np.ndarray, action: int):
        tensor = self._create_state_action_vector(state, action)
        return self.target_net(tensor)

    def get_max_value_action(self, state: np.ndarray) -> torch.Tensor:
        q_values = []
        for action in range(2):
            q_values.append(self.calculate_target_q_value(state, action))
        _, max_value = max(enumerate(q_values), key=operator.itemgetter(1))
        return max_value

    def _get_max_value_action(self, state: np.ndarray) -> int:
        q_values = []
        for action in range(2):
            q_values.append(self.calculate_prediction_q_value(state, action))
        action, _ = max(enumerate(q_values), key=operator.itemgetter(1))
        return action

    def epsilon_greedy(self, state: np.ndarray, step_number: int = 1) -> int:
        if random.random() < self.epsilon / step_number:
            return round(random.random())
        else:
            state = self._normalize(state)
            action = self._get_max_value_action(state)
            return action

    def save_weights(self):
        torch.save(self.prediction_net.state_dict(), "../weights/cart_pole_ann.pth")

    def _create_state_action_vector(self, state: np.ndarray, action: int):
        action_tensor = torch.zeros(2)
        action_tensor[action] = 1
        state_ = torch.from_numpy(state)
        tensor = torch.cat((state_, action_tensor), dim=0)
        return tensor

    def _normalize(self, state: np.ndarray):
        state[0] = (state[0] + 4.8) / 9.6
        state[1] /= np.finfo(np.float64).max
        state[2] = (state[0] + 0.418) / 0.836
        state[3] /= np.finfo(np.float64).max
        return state

    @contextmanager
    def enable_train_mode(self):
        try:
            self.prediction_net = self.prediction_net.train(True)
            yield
        finally:
            self.prediction_net = self.prediction_net.train(False)


def train(env: gym.Env, agent: CartPoleAgent, num_episodes=20000):
    optimizer = optim.Adam(agent.prediction_net.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    with agent.enable_train_mode():
        rewards = []
        iters = []
        losses = []
        for iter_num in tqdm.tqdm(range(1, num_episodes + 1)):
            if iter_num % 50 == 0:
                agent.target_net = copy.deepcopy(agent.prediction_net)
            done = False
            env.reset()
            env.state = np.asarray([random.uniform(-0.05, 0.05) for _ in range(4)])
            state = env.state
            total_reward = 0
            while not done:
                # environment interaction
                optimizer.zero_grad()
                action = agent.epsilon_greedy(state)
                old_state = state.copy()
                state, reward, done, _ = env.step(action)
                best_next_value = agent.get_max_value_action(old_state)

                # if type(env.steps_beyond_done) == int:
                #    done = env.steps_beyond_done > 100
                # creating the target
                if done:
                    y_target = reward
                else:
                    y_target = reward + agent.gamma * best_next_value

                y_target = torch.FloatTensor([y_target])
                total_reward += reward
                loss = criterion(
                    agent.calculate_prediction_q_value(old_state, action), y_target
                )
                loss.backward()
                optimizer.step()

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
                action = agent._get_max_value_action(state)
                state, reward, _, info = env.step(action)
                if type(env.steps_beyond_done) == int:
                    done = env.steps_beyond_done > 100
                time.sleep(1 / 60)
    except KeyboardInterrupt:
        env.close()


def main():
    environment = gym.make("CartPole-v0")
    agent = CartPoleAgent()
    # demonstrate(environment, agent)
    # train(environment, agent)
    # agent.save_weights()
    agent.prediction_net.load_state_dict(torch.load("../weights/cart_pole_ann.pth"))
    agent.target_net.load_state_dict(torch.load("../weights/cart_pole_ann.pth"))
    demonstrate(environment, agent)


if __name__ == "__main__":
    main()
