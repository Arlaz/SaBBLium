# coding=utf-8
#
# Copyright © Facebook, Inc. and its affiliates.
# Copyright © Sorbonne University
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import copy
import pickle
import time
from abc import ABC

import torch
import torch.nn as nn


def load(filename):
    """Load the agent from a file

    Args:
        filename (str): The filename to use
    """
    try:
        with open(filename, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        raise Exception("Could not load agent from file {filename} because of {e}".format(filename=filename, e=e))


class Agent(nn.Module, ABC):
    """An `Agent` is a `torch.nn.Module` that reads and writes into a `sabblium.Workspace`"""

    def __init__(self, name: str = None, *args, **kwargs):
        """To create a new Agent

        Args:
            name ([type], optional): An agent can have a name that will allow to perform operations on agents that are composed into more complex agents.
        """
        super().__init__(*args, **kwargs)
        self.workspace = None
        self._name = name
        self.__trace_file = None

    def set_name(self, name):
        """Set the name of this agent

        Args:
            name (str): The name
        """
        self._name = name

    def get_name(self):
        """Get the name of the agent

        Returns:
            str: the name
        """
        return self._name

    def set_trace_file(self, filename):
        print("[TRACE]: Tracing agent in file " + filename)
        self.__trace_file = open(filename, "wt")

    def __call__(self, workspace, **kwargs):
        """Execute an agent of a `sabblium.Workspace`

        Args:
            workspace (sabblium.Workspace): the workspace on which the agent operates.
        """
        assert workspace is not None, "[{}.__call__] workspace must not be None".format(self.__name__)
        self.workspace = workspace
        self.forward(**kwargs)
        self.workspace = None

    def forward(self, **kwargs):
        """The generic function to override when defining a new agent"""
        raise NotImplementedError('Your agent must override forward')

    def clone(self):
        """Create a clone of the agent

        Returns:
            sabblium.Agent: A clone
        """
        self.zero_grad()
        return copy.deepcopy(self)

    def get(self, index):
        """Returns the value of a particular variable in the agent workspace

        Args:
            index (str or tuple(str,int)): if str, returns the variable workspace[str].
            If tuple(var_name,t), returns workspace[var_name] at time t
        """
        if self.__trace_file is not None:
            t = time.time()
            self.__trace_file.write(
                str(self) + " type = " + str(type(self)) + " time = ",
                t,
                " get ",
                index,
                "\n",
            )
        if isinstance(index, str):
            return self.workspace.get_full(index)
        else:
            return self.workspace.get(index[0], index[1])

    def get_time_truncated(self, var_name, from_time, to_time):
        """Return a variable truncated between from_time and to_time"""
        return self.workspace.get_time_truncated(var_name, from_time, to_time)

    def set(self, index, value):
        """Write a variable in the workspace

        Args:
            index (str or tuple(str,int)):
            value (torch.Tensor): the value to write.
        """
        if self.__trace_file is not None:
            t = time.time()
            self.__trace_file.write(
                str(self) + " type = " + str(type(self)) + " time = ",
                t,
                " set ",
                index,
                " = ",
                value.size(),
                "/",
                value.dtype,
                "\n",
            )
        if isinstance(index, str):
            self.workspace.set_full(index, value)
        else:
            self.workspace.set(index[0], index[1], value)

    def get_by_name(self, n):
        """Returns the list of agents included in this agent that have a particular name."""
        if n == self._name:
            return [self]
        return []


class TimeAgent(Agent, ABC):
    """
    `TAgent` is used as a convention to represent agents that
    use a time index in their `__call__` function (not mandatory)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, t, *args, **kwargs):
        raise NotImplementedError('Your TAgent must override forward with a time index')


class SerializableAgent(Agent, ABC):
    """
    `SerializableAgent` is used as a convention to represent agents that are serializable (not mandatory)
    You should override the serialize method to return a dict of the parameters of the agent
    You don't have to take care of the nn.Module parameters
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def serialize(self):
        """
        Return the agent without the unsersializable attributes
        """
        try:
            return self
        except Exception as e:
            raise Exception("Could not serialize your {c} SerializableAgent because of {e}\n"
                            "You have to override the serialize method".format(c=self.__class__.__name__, e=e))

    def save(self, filename):
        """Save the agent to a file

        Args:
            filename (str): The filename to use
        """
        try:
            with open(filename, 'wb') as f:
                pickle.dump(self.serialize(), f, pickle.DEFAULT_PROTOCOL)
        except Exception as e:
            raise Exception("Could not save agent to file {filename} because of {e}"
                            "Make sure to have properly overriden the serialize method.".format(filename=filename, e=e))
