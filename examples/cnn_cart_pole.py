import gym
from torch.nn.modules.activation import LeakyReLU
import tqdm
import time
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from typing import List, Any
from collections import deque
from contextlib import contextmanager
from torchvision.transforms import ToTensor, Resize
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
        self.memory = deque(maxlen=size)

    def append(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.memory.append(experience)

    def sample(self, batch_size: int = 32):
        return self.gather(random.sample(self.memory, batch_size))

    def is_full(self) -> bool:
        return self.memory.maxlen == len(self.memory)

    def gather(self, sample: List[Any]):
        state, action, reward, n_state, done = [], [], [], [], []
        for elem in sample:
            state.append(elem[0])
            action.append(elem[1])
            reward.append(elem[2])
            n_state.append(elem[3])
            done.append(elem[4])
        return state, action, reward, n_state, done


class CartPoleAgent:
    state = None
    q_values = None
    next_q_values = None
    action = None
    writer = SummaryWriter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    to_tensor = ToTensor()
    resize = Resize(100)

    def __init__(self, epsilon: float = 0.1, gamma: float = 0.9):
        self.policy_net = self.create_neural_net().to(self.device)
        self.epsilon = epsilon
        self.gamma = gamma

    def create_neural_net(self) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=0),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, 3, padding=0),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 16, 3, padding=0),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(216576, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 2),
        )

    def normalize(self, state):
        if type(state) != torch.Tensor:
            tensor = torch.from_numpy(state)
        else:
            tensor = state.clone()
        tensor = tensor.transpose(-1, -2).transpose(-2, -3)
        tensor = self.resize(tensor)
        tensor /= 255.0
        return tensor.float()

    def get_policy_result(self, state: np.ndarray) -> torch.Tensor:
        tensor = self.normalize(state)
        return self.policy_net(tensor.to(self.device))

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

    def save_weights(self, path: str):
        torch.save(self.policy_net.state_dict(), path)

    @contextmanager
    def enable_train_mode(self):
        try:
            self.policy_net = self.policy_net.train(True)
            yield
        finally:
            self.policy_net = self.policy_net.train(False)


def train(
    env: gym.Env,
    policy_agent: CartPoleAgent,
    target_agent: CartPoleAgent,
    num_episodes=10000,
):
    optimizer = optim.Adam(
        policy_agent.policy_net.parameters(), lr=0.001, weight_decay=0.01
    )
    criterion = nn.MSELoss(reduction="none")
    target_agent.policy_net = target_agent.policy_net.train(False)
    try:
        with policy_agent.enable_train_mode():
            losses = [0]
            bar = tqdm.tqdm(range(1, num_episodes + 1))
            memory = ReplayMemory(300)
            for iter_num in bar:
                if iter_num % 40 == 0:
                    target_agent.policy_net.load_state_dict(
                        policy_agent.policy_net.state_dict()
                    )
                done = False
                env.reset()
                state = env.render("rgb_array").astype(np.float64)
                total_reward = 0
                best_reward = 0
                while not done:
                    action = policy_agent.epsilon_greedy(np.expand_dims(state, axis=0))
                    old_state = state.copy()
                    info, reward, done, _ = env.step(action)
                    state = env.render("rgb_array").astype(np.float64)
                    if done:
                        reward = -1
                    elif np.abs(info[0]) > 2:
                        reward = 0
                    total_reward += reward
                    memory.append(
                        torch.from_numpy(old_state),
                        action,
                        reward,
                        torch.from_numpy(state),
                        int(done),
                    )

                if memory.is_full():
                    states, actions, rewards, n_states, dones = memory.sample(10)
                    states = torch.stack(states, 0)
                    actions = torch.FloatTensor(actions).unsqueeze(dim=1).long()
                    rewards = torch.FloatTensor(rewards)
                    n_states = torch.stack(n_states, 0)
                    dones = torch.FloatTensor(dones)
                    with torch.no_grad():
                        max_next_q, _ = torch.max(
                            target_agent.get_policy_result(n_states.float()), dim=-1
                        )

                    targets = (
                        rewards + (1 - dones) * policy_agent.gamma * max_next_q.cpu()
                    ).unsqueeze(dim=1)
                    prediction = policy_agent.get_policy_result(states.float()).gather(
                        1, actions.cuda()
                    )

                    loss = criterion(prediction, targets.cuda())
                    loss.backward(loss)
                    losses.append(loss.mean().detach().item())
                    optimizer.step()
                    optimizer.zero_grad()
                    policy_agent.writer.add_scalar("loss", loss.mean().item(), iter_num)
                    policy_agent.writer.add_scalar(
                        "total_reward", total_reward / 200, iter_num
                    )
                    bar.set_description(
                        "Loss -> {:1.6f} Reward -> {:1.3f}".format(
                            loss.mean().item(), total_reward / 200
                        )
                    )
                    if total_reward >= best_reward:
                        policy_agent.save_weights("../weights/cart_pole_cnn.pth")
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
            env.reset()
            state = env.render("rgb_array").astype(np.float64)
            while not done:
                action = torch.argmax(
                    agent.get_policy_result(np.expand_dims(state, axis=0))
                ).item()
                _, reward, _done, info = env.step(action)
                state = env.render("rgb_array").astype(np.float64)
                env.render()
                if type(env.steps_beyond_done) == int:
                    done = env.steps_beyond_done > 100
                time.sleep(1 / 60)
    except KeyboardInterrupt:
        env.close()


def main():
    environment = gym.make("CartPole-v0")
    policy_agent = CartPoleAgent()
    target_agent = CartPoleAgent()
    train(environment, policy_agent, target_agent)
    demonstrate(environment, policy_agent)


if __name__ == "__main__":
    main()
