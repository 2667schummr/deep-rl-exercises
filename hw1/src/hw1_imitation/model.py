"""Model definitions for Push-T imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
from torch import nn


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        """Compute training loss for a batch."""

    @abc.abstractmethod
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,  # only applicable for flow policy
    ) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""


class MSEPolicy(BasePolicy):
    """Predicts action chunks with an MSE loss."""

    ### TODO: IMPLEMENT MSEPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)

        layers = [nn.Linear(state_dim, hidden_dims[0]), nn.ReLU()]
        for j in range(1, len(hidden_dims)):
            layers.append(
                nn.Linear(hidden_dims[j - 1], hidden_dims[j])
            )
            layers.append(nn.ReLU())
        
        layers.append(
            nn.Linear(hidden_dims[-1], action_dim*chunk_size)
        )
        
        self.mlp = nn.Sequential(*layers)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        action_chunk_pred = self.sample_actions(state)
        
        loss = (action_chunk - action_chunk_pred)**2
        loss = loss.sum(dim=(1, 2))
        loss = loss.mean()

        return loss

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        action_chunk = self.mlp(state).reshape(
            state.shape[0], self.chunk_size, self.action_dim
        )

        return action_chunk


class FlowMatchingPolicy(BasePolicy):
    """Predicts action chunks with a flow matching loss."""

    ### TODO: IMPLEMENT FlowMatchingPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        
        layers = [
            nn.Linear(
                state_dim + action_dim*chunk_size + 1,
                hidden_dims[0]
            ), 
            nn.ReLU()
        ]
        for j in range(1, len(hidden_dims)):
            layers.append(
                nn.Linear(hidden_dims[j - 1], hidden_dims[j])
            )
            layers.append(nn.ReLU())
        
        layers.append(
            nn.Linear(hidden_dims[-1], action_dim*chunk_size)
        )
        
        self.mlp = nn.Sequential(*layers)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = action_chunk.shape[0]
        
        action_chunk_0 = torch.randn(
            (batch_size, self.action_dim*self.chunk_size),
            device=state.device
        )
        taus = torch.rand(
            (batch_size, 1), device=state.device
        )
        action_chunk_tau = taus*action_chunk.reshape(
            batch_size, self.chunk_size*self.action_dim
        )
        action_chunk_tau += (1.0 - taus)*action_chunk_0
        nn_input = torch.concat(
            (state, action_chunk_tau, taus), dim=1
        )
        
        velocities_pred = self.mlp(nn_input)
        velocities_true = action_chunk.reshape(batch_size, -1)
        velocities_true -= action_chunk_0


        loss = (velocities_pred - velocities_true)**2
        loss = loss.sum(dim=-1)
        loss = loss.mean()

        return loss

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        batch_size = state.shape[0]
        action_chunk = torch.randn(
            (batch_size, self.action_dim*self.chunk_size),
            device=state.device
        )
        times = torch.zeros((batch_size, 1), device=state.device)

        for time in torch.linspace(0.0, 1.0, steps=num_steps):
            times[:, 0] = time
            nn_input = torch.concat(
                (state, action_chunk, times), dim=1
            )
            action_chunk += self.mlp(nn_input) / num_steps 
        
        action_chunk = action_chunk.reshape(
            batch_size, self.chunk_size, self.action_dim
        )
        
        return action_chunk


PolicyType: TypeAlias = Literal["mse", "flow"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    hidden_dims: tuple[int, ...] = (128, 128),
) -> BasePolicy:
    if policy_type == "mse":
        return MSEPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    if policy_type == "flow":
        return FlowMatchingPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
