# coding=utf-8
#
# Copyright © Facebook, Inc. and its affiliates.
# Copyright © Sorbonne University
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
from abc import ABC
from typing import List, Optional, Tuple

import torch


class TemporalTensor(ABC):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.size: Optional[torch.Size] = None
        self.device: Optional[torch.device] = None
        self.dtype: Optional[torch.dtype] = None

    def clear(self) -> None:
        self.size = None
        self.device = None
        self.dtype = None

    def zero_grad(self) -> None:
        raise NotImplementedError("zero_grad is not implemented")

    def time_size(self) -> int:
        raise NotImplementedError("time_size is not implemented")

    def batch_size(self) -> int:
        raise NotImplementedError("batch_size is not implemented")

    def get(self, t: int, batch_dims: Optional[Tuple[int, int]]) -> torch.Tensor:
        raise NotImplementedError("get is not implemented")

    def get_full(self, batch_dims: Optional[Tuple[int, int]]) -> torch.Tensor:
        raise NotImplementedError("get_full is not implemented")

    def set(
        self,
        t: int,
        value: torch.Tensor,
        batch_dims: Optional[Tuple[int, int]],
    ) -> None:
        raise NotImplementedError("set is not implemented")

    def set_full(
        self, value: torch.Tensor, batch_dims: Optional[Tuple[int, int]]
    ) -> None:
        raise NotImplementedError("set_full is not implemented")

    def select_batch(self, batch_indexes: torch.Tensor) -> "TemporalTensor":
        raise NotImplementedError("select_batch is not implemented")

    def to_sliced(self) -> "SlicedTemporalTensor":
        if isinstance(self, SlicedTemporalTensor):
            return self
        else:
            raise NotImplementedError("to_sliced is not implemented")

    def copy_time(self, from_time: int, to_time: int, n_steps: int) -> None:
        raise NotImplementedError("copy_time is not implemented")

    def subtime(self, from_t: int, to_t: int):
        raise NotImplementedError("subtime is not implemented")

    def to(self, device: torch.device) -> "TemporalTensor":
        raise NotImplementedError("to is not implemented")


class SlicedTemporalTensor(TemporalTensor):
    """A `SlicedTemporalTensor` represents a tensor of size (T × B × …) by using a list of tensors of size (B × …)
    The interest is that this tensor automatically adapts its timestep dimension
    and does not need to have a predefined size.
    """

    def __init__(self, *args, **kwargs) -> None:
        """Initialize an empty tensor"""
        super().__init__(*args, **kwargs)
        self.tensors: List[torch.Tensor] = []
        self.size: Optional[torch.Size] = None
        self.device: Optional[torch.device] = None
        self.dtype: Optional[torch.dtype] = None

    def set(
        self,
        t: int,
        value: torch.Tensor,
        batch_dims: Optional[Tuple[int, int]],
    ) -> None:
        """Set a value of size (B × …) at time t"""
        if batch_dims is not None:
            raise TypeError("Unable to use batch dimensions with SlicedTemporalTensor")

        if self.size is None:
            self.size = value.size()
            self.device = value.device
            self.dtype = value.dtype
        assert self.size == value.size(), "Incompatible size"
        assert self.device == value.device, "Incompatible device"
        assert self.dtype == value.dtype, "Incompatible type"
        while len(self.tensors) <= t:
            self.tensors.append(
                torch.zeros(*self.size, device=self.device, dtype=self.dtype)
            )
        self.tensors[t] = value

    def to(self, device: torch.device) -> "SlicedTemporalTensor":
        """Move the tensor to a specific device"""
        s: "SlicedTemporalTensor" = SlicedTemporalTensor()
        for k in range(len(self.tensors)):
            s.set(k, self.tensors[k].to(device), batch_dims=None)
        return s

    def get(self, t: int, batch_dims: Optional[Tuple[int, int]]) -> torch.Tensor:
        """Get the value of the tensor at time t"""

        assert (
            batch_dims is None
        ), "Unable to use batch dimensions with SlicedTemporalTensor"
        assert t < len(self.tensors), "Temporal index out of bounds"
        return self.tensors[t]

    def get_full(self, batch_dims) -> torch.Tensor:
        """Returns the complete tensor of size (T × B × …)"""

        assert (
            batch_dims is None
        ), "Unable to use batch dimensions with SlicedTemporalTensor"
        return torch.cat([a.unsqueeze(0) for a in self.tensors], dim=0)

    def get_time_truncated(
        self,
        from_time: int,
        to_time: int,
        batch_dims: Optional[Tuple[int, int]],
    ) -> torch.Tensor:
        """Returns tensor[from_time:to_time]"""
        assert 0 <= from_time < to_time and to_time >= 0
        assert batch_dims is None
        return torch.cat(
            [
                self.tensors[k].unsqueeze(0)
                for k in range(from_time, min(len(self.tensors), to_time))
            ],
            dim=0,
        )

    def set_full(
        self, value: torch.Tensor, batch_dims: Optional[Tuple[int, int]]
    ) -> None:
        """Set the tensor given a (T × B × …) value tensor.
        The input tensor is cut into slices that are stored in a list of tensors
        """
        assert (
            batch_dims is None
        ), "Unable to use batch dimensions with SlicedTemporalTensor"
        for t in range(value.size()[0]):
            self.set(t, value[t], batch_dims=batch_dims)

    def time_size(self) -> int:
        """
        Return the size of the time dimension
        """
        return len(self.tensors)

    def batch_size(self) -> int:
        """Return the size of the batch dimension"""
        return self.tensors[0].size()[0]

    def select_batch(self, batch_indexes: torch.Tensor) -> "SlicedTemporalTensor":
        """Return the tensor where the batch dimension has been selected by the index"""
        assert batch_indexes.dtype in (torch.short, torch.int, torch.long)
        var = SlicedTemporalTensor()
        for t, v in enumerate(self.tensors):
            batch_indexes = batch_indexes.to(v.device)
            var.set(t, v[batch_indexes], None)
        return var

    def clear(self):
        """Clear the tensor"""
        super().clear()
        self.tensors = []

    def copy_time(self, from_time: int, to_time: int, n_steps: int):
        """Copy temporal slices of the tensor from from_time:from_time+n_steps to to_time:to_time+n_steps"""
        for t in range(n_steps):
            v = self.get(from_time + t, batch_dims=None)
            self.set(to_time + t, v, batch_dims=None)

    def subtime(self, from_t: int, to_t: int) -> "CompactTemporalTensor":
        """
        Return tensor[from_t:to_t]

        """
        return CompactTemporalTensor(
            value=torch.cat([a.unsqueeze(0) for a in self.tensors[from_t:to_t]], dim=0),
        )

    def zero_grad(self) -> None:
        """Clear any gradient information in the tensor"""
        self.tensors = [v.detach() for v in self.tensors]


class CompactTemporalTensor(TemporalTensor):
    """
    A `CompactTemporalTensor` is a tensor of size (T × B × …)
    It behaves like the `SlicedTemporalTensor` but has a fixed size that cannot change.
    It is faster than the `SlicedTemporalTensor`.
    """

    def __init__(self, value: Optional[torch.Tensor] = None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.size = None
        self.device = None
        self.dtype = None
        self.tensor = None
        if value is not None:
            self.tensor = value
            self.device = value.device
            self.size = value.size()
            self.dtype = value.dtype

    def set(self, t, value, batch_dims) -> SlicedTemporalTensor:
        raise NotImplementedError("Cannot set a value in a CompactTemporalTensor")

    def select_batch(self, batch_indexes: torch.Tensor) -> "CompactTemporalTensor":
        assert batch_indexes.dtype in (torch.short, torch.int, torch.long)
        return CompactTemporalTensor(value=self.tensor[:, batch_indexes])

    def to_sliced(self) -> SlicedTemporalTensor:
        """Transform the tensor to a `SlicedTemporalTensor`"""
        v = SlicedTemporalTensor()
        for t in range(self.tensor.size()[0]):
            v.set(t, self.tensor[t], None)
        return v

    def to(self, device: torch.device) -> "CompactTemporalTensor":
        if device == self.tensor.device:
            return self
        return CompactTemporalTensor(value=self.tensor.to(device))

    def get(self, t: int, batch_dims: Optional[Tuple[int, int]]):
        assert t < self.tensor.size()[0], "Temporal index out of bounds"
        if batch_dims is None:
            return self.tensor[t]
        else:
            return self.tensor[t, batch_dims[0] : batch_dims[1]]

    def get_full(self, batch_dims: Optional[Tuple[int, int]]):
        if batch_dims is None:
            return self.tensor
        else:
            return self.tensor[:, batch_dims[0] : batch_dims[1]]

    def time_size(self) -> int:
        return self.tensor.size()[0]

    def batch_size(self) -> int:
        return self.tensor.size()[1]

    def set_full(
        self,
        value: Optional[torch.Tensor],
        batch_dims: Optional[Tuple[int, int]],
    ):
        if self.tensor is None:
            if batch_dims is None:
                self.size = value.size()
                self.dtype = value.dtype
                self.device = value.device
            else:
                raise TypeError("Cannot set an empty tensor with batch_dims")

        if batch_dims is None:
            if value is not None:
                self.tensor = value
            else:
                raise TypeError("Value is None but batch_dims is not")
        else:
            self.tensor[:, batch_dims[0] : batch_dims[1]] = value

    def subtime(self, from_t: int, to_t: int):
        return CompactTemporalTensor(value=self.tensor[from_t:to_t])

    def clear(self):
        super().clear()
        self.tensor = None

    def copy_time(self, from_time: int, to_time: int, n_steps: int):
        self.tensor[to_time : to_time + n_steps] = self.tensor[
            from_time : from_time + n_steps
        ]

    def zero_grad(self):
        self.tensor = self.tensor.detach()
