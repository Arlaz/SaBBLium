# coding=utf-8
#
# Copyright © Facebook, Inc. and its affiliates.
# Copyright © Sorbonne University
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn

from sabblium import Agent, TimeAgent, SerializableAgent
from sabblium.agents.seeding import SeedableAgent


class Agents(SeedableAgent, SerializableAgent):
    """An agent that contains multiple agents executed sequentially.
    Warnings:
        * the agents are executed in the order they are added to the agent.
        * the agents are serialized only if they inherit from `SerializableAgent`
        * the agents are seeded only if they inherit from `SeedableAgent`, with the same seed provided

    Args:
        Agent ([sabblium.Agent]): The agents
    """

    def __init__(self, *agents, name=None, **kwargs):
        """Creates the agent from multiple agents

        Args:
            name ([str], optional): [name of the resulting agent]. Default to None.
        """
        super().__init__(name=name, **kwargs)
        for a in agents:
            assert isinstance(a, Agent)
        self.agents = nn.ModuleList(agents)

    def __getitem__(self, k):
        return self.agents[k]

    def __call__(self, workspace, **kwargs):
        for a in self.agents:
            a(workspace, **kwargs)

    def get_by_name(self, n):
        r = []
        for a in self.agents:
            r += a.get_by_name(n)
        if n == self._name:
            r += [self]
        return r

    def seed(self, seed: int):
        """Seed the agents
        Warning: the agents are seeded  with the same seed and only if they inherit from `SeedableAgent`
        Args:
            seed (int): the seed to use
        """
        for a in self.agents:
            if isinstance(a, SeedableAgent):
                a.seed(seed)

    def serialize(self):
        """Serialize the agents
        Warning: the agents are serialized only if they inherit from `SerializableAgent`
        Args:
            filename (str): the filename to use
        """
        serializable_agents = [
            a.serialize() if isinstance(a, SerializableAgent) else (a.__class__.__name__, a.get_name())
            for a in self.agents
        ]
        return Agents(*serializable_agents, name=self._name)


class CopyTAgent(SerializableAgent):
    """An agent that copies a variable

    Args:
        input_name ([str]): The variable to copy from
        output_name ([str]): The variable to copy to
        detach ([bool]): copy with detach if True
    """

    def __init__(self, input_name, output_name, detach=False, name=None):
        super().__init__(name=name)
        self.input_name = input_name
        self.output_name = output_name
        self.detach = detach

    def forward(self, t=None, **kwargs):
        """
        Args:
            t ([type], optional): if not None, copy at time t. Defaults to None.
        """
        if t is None:
            x = self.get(self.input_name)
            if not self.detach:
                self.set(self.output_name, x)
            else:
                self.set((self.output_name, t), x.detach())
        else:
            x = self.get((self.input_name, t))
            if not self.detach:
                self.set((self.output_name, t), x)
            else:
                self.set((self.output_name, t), x.detach())


class PrintAgent(SerializableAgent):
    """An agent to generate print in the console (mainly for debugging)

    Args:
        Agent ([type]): [description]
    """

    def __init__(self, *names, name=None):
        """
        Args:
            names ([str], optional): The variables to print
        """
        super().__init__(name=name)
        self.names = names

    def forward(self, t, **kwargs):
        for n in self.names:
            print(n, " = ", self.get((n, t)))


class TemporalAgent(TimeAgent, SeedableAgent, SerializableAgent):
    """Execute one Agent over multiple timestamps

    Args:
        Agent ([sabblium.Agent])
    """

    def __init__(self, agent, name=None):
        """The agent to transform to a temporal agent

        Args:
            agent ([sabblium.Agent]): The agent to encapsulate
            name ([str], optional): Name of the agent
        """
        super().__init__(name=name)
        self.agent = agent

    def __call__(self, workspace, t=0, n_steps=None, stop_variable=None, **kwargs):
        """Execute the agent starting at time t, for n_steps

        Args:
            workspace ([sabblium.Workspace]):
            t (int, optional): The starting timestep. Defaults to 0.
            n_steps ([type], optional): The number of steps. Defaults to None.
            stop_variable ([type], optional): if True everywhere (at time t), execution is stopped. Defaults to None = not used.
        """

        assert not (n_steps is None and stop_variable is None)
        _t = t
        while True:
            self.agent(workspace, t=_t, **kwargs)
            if stop_variable is not None:
                s = workspace.get(stop_variable, _t)
                if s.all():
                    break
            _t += 1
            if n_steps is not None:
                if _t >= t + n_steps:
                    break

    def get_by_name(self, n):
        r = self.agent.get_by_name(n)
        if n == self._name:
            r += [self]
        return r

    def seed(self, seed: int):
        """Seed the agent

        Args:
            seed: int: the seed to use
        """
        self.agent.seed(seed)

    def serialize(self):
        """Can only serialize if the wrapped agent is serializable"""
        if isinstance(self.agent, SerializableAgent):
            return TemporalAgent(self.agent.serialize(), name=self._name)
        else:
            return TemporalAgent(None, name=self._name)


class EpisodesDone(TimeAgent, SerializableAgent):
    """
    If stopped is encountered at time t, then stopped=True for all timeteps t'>=t
    It allows to simulate a single episode agent based on an autoreset agent
    """

    def __init__(self, in_var="env/stopped", out_var="env/stopped"):
        super().__init__()
        self.in_var = in_var
        self.out_var = out_var

    def forward(self, t, **kwargs):
        d = self.get((self.in_var, t))
        if t == 0:
            self.state = torch.zeros_like(d).bool()
        self.state = torch.logical_or(self.state, d)
        self.set((self.out_var, t), self.state)
