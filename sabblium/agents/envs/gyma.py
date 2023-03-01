# coding=utf-8
#
# Copyright © Facebook, Inc. and its affiliates.
# Copyright © Sorbonne University
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import gymnasium as gym
import numpy as np
import torch

from sabblium import TAgent


def _convert_action(action):
    if len(action.size()) == 0:
        action = action.item()
        assert isinstance(action, int)
    else:
        action = np.array(action.tolist())
    return action


def _format_frame(frame):
    if isinstance(frame, dict):
        r = {}
        for k in frame:
            r[k] = _format_frame(frame[k])
        return r
    elif isinstance(frame, list):
        t = torch.tensor(frame).unsqueeze(0)
        if t.dtype == torch.float64 or t.dtype == torch.float32:
            t = t.float()
        else:
            t = t.long()
        return t
    elif isinstance(frame, np.ndarray):
        t = torch.from_numpy(frame).unsqueeze(0)
        if t.dtype == torch.float64 or t.dtype == torch.float32:
            t = t.float()
        else:
            t = t.long()
        return t
    elif isinstance(frame, torch.Tensor):
        return frame.unsqueeze(0)  # .float()
    elif isinstance(frame, bool):
        return torch.tensor([frame]).bool()
    elif isinstance(frame, int):
        return torch.tensor([frame]).long()
    elif isinstance(frame, float):
        return torch.tensor([frame]).float()

    else:
        try:
            # Check if it is a LazyFrame from OpenAI Baselines
            o = torch.from_numpy(frame.__array__()).unsqueeze(0).float()
            return o
        except TypeError:
            assert False


def _torch_type(d):
    nd = {}
    for k in d:
        if d[k].dtype == torch.float64:
            nd[k] = d[k].float()
        else:
            nd[k] = d[k]
    return nd


def _torch_cat_dict(d):
    r = {}
    for k in d[0]:
        r[k] = torch.cat([dd[k] for dd in d], dim=0)
    return r


class GymAgent(TAgent):
    """Create an Agent from a gym environment
    To create an AutoResetGymAgent, use the gymnasium AutoResetWrapper before creating the GymAgent
    """

    def __init__(
        self,
        make_env_fn=None,
        make_env_args={},
        n_envs=None,
        input_string="action",
        output_string="env/",
        use_seed=True,
    ):
        """Create an agent from a Gym environment

        Args:
            make_env_fn ([function that returns a gym.Env]): The function to create a single gym environments
            make_env_args (dict): The arguments of the function that creates a gym.Env
            n_envs ([int]): The number of environments to create.
            input_string (str, optional): [the name of the action variable in the workspace]. Defaults to "action".
            output_string (str, optional): [the output prefix of the environment]. Defaults to "env/".
            use_seed (bool, optional): [If True, then the seed is chained to the environments, and each environment will have its own seed]. Defaults to True.
        """
        super().__init__()
        self.timestep = torch.zeros(len(self.envs), dtype=torch.int64)
        self.timestep = None
        assert n_envs > 0, "n_envs must be > 0"

        self.make_env_fn = make_env_fn
        self.env_args = make_env_args
        self.n_envs = n_envs
        self.input = input_string
        self.output = output_string
        self.use_seed = use_seed

        self._seed = 0
        self.ghost_params = torch.nn.Parameter(torch.randn(()))

        self.envs = None

    def _initialize_envs(self, n):
        assert self._seed is not None, "[GymAgent] seeds must be specified"
        self.envs = [self.make_env_fn(**self.env_args) for _ in range(n)]
        self.timestep = torch.zeros(len(self.envs), dtype=torch.int64)
        self.cumulated_reward = {}
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

    def _reset(self, k, render):
        env = self.envs[k]
        self.cumulated_reward[k] = 0.0
        o, info = env.reset(seed=self._seed + k)
        observation = _format_frame(o)

        if isinstance(observation, torch.Tensor):
            observation = {"env_obs": observation}
        else:
            assert isinstance(observation, dict)
        if render:
            observation["rendering"] = env.render().unsqueeze(0)

        self.timestep[k] = 0

        ret = {
            **observation,
            "terminated": torch.tensor([False]),
            "truncated": torch.tensor([False]),
            "reward": torch.tensor([0.0]).float(),
            "cumulated_reward": torch.tensor(
                [self.cumulated_reward[k]]
            ).float(),
            "timestep": torch.tensor([self.timestep[k]]),
        }
        return _torch_type(ret)

    def _step(self, k, action, render):
        env = self.envs[k]
        action = _convert_action(action)

        obs, reward, terminated, truncated, info = env.step(action)

        self.cumulated_reward[k] += reward
        observation = _format_frame(obs)

        if isinstance(observation, torch.Tensor):
            observation = {"env_obs": observation}
        else:
            assert isinstance(observation, dict)
        if render:
            observation["rendering"] = env.render().unsqueeze(0)

        ret = {
            **observation,
            "terminated": torch.tensor([terminated]),
            "truncated": torch.tensor([truncated]),
            "reward": torch.tensor([reward]).float(),
            "cumulated_reward": torch.tensor([self.cumulated_reward[k]]),
            "timestep": torch.tensor([self.timestep[k]]),
        }
        return _torch_type(ret)

    def set_obs(self, observations, t):
        observations = _torch_cat_dict(observations)
        for k in observations:
            self.set(
                (self.output + k, t),
                observations[k].to(self.ghost_params.device),
            )

    def forward(self, t=0, render=False, **kwargs):
        """Do one step by reading the `action` at t-1
        If t==0, environments are reset
        If render is True, then the output of env.render() is written as env/rendering
        """
        if self.envs is None:
            self._initialize_envs(self.n_envs)

        observations = []
        if t == 0:
            for k, env in enumerate(self.envs):
                observations.append(self._reset(k, render))
        else:
            action = self.get((self.input, t - 1))
            assert (
                action.size()[0] == self.n_envs
            ), "Incompatible number of envs"
            for k, env in enumerate(self.envs):
                observations.append(self._step(k, action[k], render))
        self.set_obs(observations, t)

    def seed(self, seed: int):
        """Set the seed of the environments
        Will only take effect if called before the environments are created
        """
        assert (
            self.envs is None
        ), "Cannot set seed after environments have been created"
        if self.use_seed:
            self._seed = seed
        else:
            raise ValueError("Cannot set seed if use_seed is False")

    def is_continuous_action(self):
        return isinstance(self.action_space, gym.spaces.Box)

    def is_discrete_action(self):
        return isinstance(self.action_space, gym.spaces.Discrete)

    def is_continuous_state(self):
        return isinstance(self.observation_space, gym.spaces.Box)

    def is_discrete_state(self):
        return isinstance(self.observation_space, gym.spaces.Discrete)

    def get_obs_and_actions_sizes(self):
        if self.envs is None:
            raise ValueError("Environments have not been created yet")
        state_dim, action_dim = 0, 0

        if self.is_continuous_state():
            state_dim = self.observation_space.shape[0]
        elif self.is_discrete_state():
            state_dim = 1  # self.observation_space.n

        if self.is_continuous_action():
            action_dim = self.action_space.shape[0]
        elif self.is_discrete_action():
            action_dim = self.action_space.n

        return state_dim, action_dim
