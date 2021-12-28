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
from torchvision.transforms import ToTensor
from torch.utils.tensorboard import SummaryWriter


"""
Mountain Car System Notes

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


class MountainCarAgent:
    state = None
    q_values = None
    next_q_values = None
    action = None
    writer = SummaryWriter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    to_tensor = ToTensor()

    def __init__(self, epsilon: float = 0.1, gamma: float = 0.9):
        self.policy_net = self.create_neural_net().to(self.device)
        self.epsilon = epsilon
        self.gamma = gamma

    def create_neural_net(self) -> nn.Module:
        return nn.Sequential(
            nn.Linear(2, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 3),
        )

    def normalize(self, state):
        if type(state) != torch.Tensor:
            tensor = torch.from_numpy(state)
        else:
            tensor = state.clone()
        tensor[..., 0] /= 1.8
        tensor[..., 1] /= 0.14
        return tensor

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
    env: gym.Env, policy_agent: MountainCarAgent, num_episodes=10000,
):
    optimizer = optim.Adam(
        policy_agent.policy_net.parameters(), lr=0.001, weight_decay=0.01
    )
    criterion = nn.MSELoss(reduction="none")

    try:
        with policy_agent.enable_train_mode():
            losses = [0]
            bar = tqdm.tqdm(range(1, num_episodes + 1))
            memory = ReplayMemory(300)
            for iter_num in bar:
                done = False
                state = env.reset()
                env.render()
                total_reward = 0
                best_reward = 0
                while not done:
                    action = policy_agent.epsilon_greedy(state)
                    old_state = state.copy()
                    state, reward, done, _ = env.step(action)
                    reward = state[0]

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
                            policy_agent.get_policy_result(n_states.float()), dim=-1
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
                        policy_agent.save_weights("../weights/cart_pole_ann.pth")
                        best_reward = total_reward
        plt.plot(losses)
        plt.show()
    except KeyboardInterrupt:
        plt.plot(losses)
        plt.show()


def demonstrate(env: gym.Env, agent: MountainCarAgent):
    try:
        while True:
            done = False
            np.random.seed(random.randint(0, 10000))
            state = env.reset()
            env.render()
            while not done:
                action = torch.argmax(agent.get_policy_result(state)).item()
                state, reward, _done, info = env.step(action)
                env.render()
                time.sleep(1 / 60)
    except KeyboardInterrupt:
        env.close()


def main():
    environment = gym.make("MountainCar-v0")
    policy_agent = MountainCarAgent()
    train(environment, policy_agent)
    demonstrate(environment, policy_agent)


if __name__ == "__main__":
    main()
