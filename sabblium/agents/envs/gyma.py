# coding=utf-8
#
# Copyright © Facebook, Inc. and its affiliates.
# Copyright © Sorbonne University
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import copy

import numpy as np
import torch
from gymnasium.wrappers import AutoResetWrapper

from sabblium import SeedableAgent, SerializableAgent, TimeAgent


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
        return frame.unsqueeze(0)
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


class GymAgent(TimeAgent, SeedableAgent, SerializableAgent):
    """Create an Agent from a gymnasium environment
    To create an auto-reset GymAgent, use the gymnasium `AutoResetWrapper` before creating the `GymAgent
    """

    default_seed = 0

    def __init__(
        self,
        make_env_fn=None,
        make_env_args={},
        n_envs=None,
        input_string="action",
        output_string="env/",
        max_episode_steps=None,
        *args,
        **kwargs,
    ):
        """Create an agent from a Gymnasium environment

        Args:
            make_env_fn ([function that returns a gymnasium.Env]): The function to create a single gymnasium environments
            make_env_args (dict): The arguments of the function that creates a gymnasium.Env
            n_envs ([int]): The number of environments to create.
            input_string (str, optional): [the name of the action variable in the workspace]. Defaults to "action".
            output_string (str, optional): [the output prefix of the environment]. Defaults to "env/".
            max_episode_steps (int, optional): Max number of steps per episode. Defaults to None (never ends)
        """
        super().__init__(*args, **kwargs)
        assert n_envs > 0, "n_envs must be > 0"

        self.make_env_fn = make_env_fn
        self.env_args = make_env_args
        self.n_envs = n_envs
        self.input = input_string
        self.output = output_string

        self.ghost_params = torch.nn.Parameter(torch.randn(()))

        self.envs = None

        self._max_episode_steps = max_episode_steps
        self._timestep = None
        self._timestep_from_reset = None
        self._is_autoreset = False
        self._stopped = {}
        self._last_frame = {}
        self._nb_reset = 0

        self._initialize_envs(n_envs)

    def _initialize_envs(self, n):
        self.envs = [self.make_env_fn(**self.env_args) for _ in range(n)]
        self._timestep = torch.zeros(len(self.envs), dtype=torch.int64)
        self._timestep_from_reset = 0
        self.cumulated_reward = {}
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

        unwrapped_env = self.envs[0].unwrapped
        wrapper = self.envs[0]
        while type(wrapper) is not type(unwrapped_env):
            if type(wrapper) == AutoResetWrapper:
                self._is_autoreset = True
            wrapper = wrapper.env

        if self._is_autoreset and self._max_episode_steps is None:
            raise ValueError("AutoResetWrapper without max_episode_steps argument given will never"
                             "stop the GymAgent if wrapped with a TemporalAgent")

    def _reset(self, k, render):
        env = self.envs[k]
        self.cumulated_reward[k] = 0.0

        s = self._max_episode_steps * self.n_envs * self._nb_reset * self._seed
        s += (k+1) * (self._timestep[k].item() + 1 if self._is_autoreset else 1)
        o, info = env.reset(seed=s)
        observation = _format_frame(o)

        self._timestep[k] = 0

        if isinstance(observation, torch.Tensor):
            observation = {"env_obs": observation}
        else:
            assert isinstance(observation, dict)
        if render:
            observation["rendering"] = env.render().unsqueeze(0)

        ret = {
            **observation,
            "terminated": torch.tensor([False]),
            "truncated": torch.tensor([False]),
            "stopped": torch.tensor([False]),
            "reward": torch.tensor([0.0]).float(),
            "cumulated_reward": torch.tensor([self.cumulated_reward[k]]),
            "timestep": torch.tensor([self._timestep[k]]),
        }
        self._last_frame[k] = ret
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

        self._timestep[k] += 1

        if terminated or truncated:
            self._timestep[k] = 0

        if self._is_autoreset:
            stopped = self._timestep_from_reset + 1 >= self._max_episode_steps
        else:
            stopped = terminated or truncated

        ret = {
            **observation,
            "terminated": torch.tensor([terminated]),
            "truncated": torch.tensor([truncated]),
            "stopped": torch.tensor([stopped]),
            "reward": torch.tensor([reward]).float(),
            "cumulated_reward": torch.tensor([self.cumulated_reward[k]]),
            "timestep": torch.tensor([self._timestep[k]]),
        }
        self._last_frame[k] = ret
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

        self._timestep_from_reset += 1

        observations = []
        if t == 0:
            self._timestep_from_reset = 0
            self._nb_reset += 1
            if self._seed is None:
                self.seed(self.default_seed)
            for k, env in enumerate(self.envs):
                observations.append(self._reset(k, render))
        else:
            action = self.get((self.input, t - 1))
            assert action.size()[0] == self.n_envs, "Incompatible number of envs"

            for k, env in enumerate(self.envs):
                if self._is_autoreset or not self._last_frame[k]["stopped"]:
                    observations.append(self._step(k, action[k], render))
                else:
                    observations.append(self._last_frame[k])
        self.set_obs(observations, t)

    def get_observation_space(self):
        """Return the observation space of the environment"""
        return self.observation_space

    def get_action_space(self):
        """Return the action space of the environment"""
        return self.action_space

    def serialize(self):
        """Return a serializable GymAgent without the environments"""
        state = copy.copy(self)
        state.envs = None
        return state


class ImageGymAgent(GymAgent, SerializableAgent):
    """
    GymAgent compatible with image observations
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def serialize(self):
        """Return a serializable GymAgent without the environments"""
        state = copy.copy(self)
        state.envs = None
        return state
