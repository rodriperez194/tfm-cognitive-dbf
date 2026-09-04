"""
Phase 5 action wrappers.

This module provides Gymnasium wrappers for BeamformingEnvPhase5.

Main wrapper:
    PhaseSincosToPhaseOnlyActionWrapper

Purpose:
    Expose a smooth sine/cosine action space to the DRL agent while keeping
    the original BeamformingEnvPhase5 unchanged.

The wrapped agent action has dimension 2 * num_elements:

    action = [cos_1, ..., cos_N, sin_1, ..., sin_N]

The base environment receives the standard phase_only action:

    phase_only_action = [p_1, ..., p_N]

where:

    phase_n = atan2(sin_n, cos_n)
    p_n = phase_n / pi
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class PhaseSincosToPhaseOnlyActionWrapper(gym.Wrapper):
    """
    Convert sine/cosine phase actions into phase_only actions.

    This wrapper is intended for BeamformingEnvPhase5 when the base environment
    is configured with:

        complex_weight_mode = "phase_only"

    The wrapper exposes a smoother action space to the agent:

        wrapped action shape = (2 * num_elements,)

    with:

        action[0:num_elements]              -> cosine components
        action[num_elements:2*num_elements] -> sine components

    The wrapper converts this action into the base environment action:

        phase = atan2(sin, cos)
        phase_only_action = phase / pi
    """

    def __init__(
        self,
        env: gym.Env,
        normalize_sincos: bool = True,
        eps: float = 1e-8,
        store_last_converted_action: bool = True,
    ) -> None:
        super().__init__(env)

        self.normalize_sincos = bool(normalize_sincos)
        self.eps = float(eps)
        self.store_last_converted_action = bool(store_last_converted_action)

        self.num_elements = self._infer_num_elements_from_base_env(env)

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2 * self.num_elements,),
            dtype=np.float32,
        )

        self.last_sincos_action: np.ndarray | None = None
        self.last_phase_only_action: np.ndarray | None = None
        self.last_phase_rad: np.ndarray | None = None

    @staticmethod
    def _infer_num_elements_from_base_env(env: gym.Env) -> int:
        """
        Infer the number of array elements from the base environment action space.

        For BeamformingEnvPhase5 in phase_only mode, the base action space has:

            shape = (num_elements,)
        """

        if not hasattr(env, "action_space"):
            raise TypeError("The wrapped environment must have an action_space attribute.")

        base_action_space = env.action_space

        if not isinstance(base_action_space, spaces.Box):
            raise TypeError(
                "PhaseSincosToPhaseOnlyActionWrapper expects the base environment "
                "to have a gymnasium.spaces.Box action space."
            )

        if len(base_action_space.shape) != 1:
            raise ValueError(
                "PhaseSincosToPhaseOnlyActionWrapper expects a 1D base action space. "
                f"Received shape: {base_action_space.shape}"
            )

        num_elements = int(base_action_space.shape[0])

        if num_elements <= 0:
            raise ValueError("The inferred number of elements must be positive.")

        return num_elements

    def convert_sincos_to_phase_only(self, sincos_action: np.ndarray) -> np.ndarray:
        """
        Convert wrapped sincos action into base phase_only action.

        Input:
            sincos_action shape = (2 * num_elements,)

        Output:
            phase_only_action shape = (num_elements,)
        """

        sincos_action = np.asarray(sincos_action, dtype=np.float32).reshape(-1)

        expected_dim = 2 * self.num_elements

        if sincos_action.shape[0] != expected_dim:
            raise ValueError(
                f"Invalid sincos action dimension. Expected {expected_dim}, "
                f"received {sincos_action.shape[0]}."
            )

        cos_values = sincos_action[: self.num_elements].astype(np.float64)
        sin_values = sincos_action[self.num_elements :].astype(np.float64)

        if self.normalize_sincos:
            norm = np.sqrt(cos_values**2 + sin_values**2)
            norm = np.maximum(norm, self.eps)

            cos_values = cos_values / norm
            sin_values = sin_values / norm

        phase_rad = np.arctan2(sin_values, cos_values)
        phase_only_action = phase_rad / np.pi
        phase_only_action = np.clip(phase_only_action, -1.0, 1.0).astype(np.float32)

        if self.store_last_converted_action:
            self.last_sincos_action = sincos_action.copy()
            self.last_phase_only_action = phase_only_action.copy()
            self.last_phase_rad = phase_rad.astype(np.float32).copy()

        return phase_only_action

    def convert_phase_only_to_sincos(self, phase_only_action: np.ndarray) -> np.ndarray:
        """
        Convert a phase_only action into wrapped sincos representation.

        This is useful for supervised pretraining and debugging.

        Input:
            phase_only_action shape = (num_elements,)

        Output:
            sincos_action shape = (2 * num_elements,)
            [cos_1, ..., cos_N, sin_1, ..., sin_N]
        """

        phase_only_action = np.asarray(phase_only_action, dtype=np.float32).reshape(-1)

        if phase_only_action.shape[0] != self.num_elements:
            raise ValueError(
                f"Invalid phase_only action dimension. Expected {self.num_elements}, "
                f"received {phase_only_action.shape[0]}."
            )

        phase_rad = np.pi * phase_only_action.astype(np.float64)

        cos_values = np.cos(phase_rad)
        sin_values = np.sin(phase_rad)

        sincos_action = np.concatenate([cos_values, sin_values], axis=0)
        sincos_action = np.clip(sincos_action, -1.0, 1.0).astype(np.float32)

        return sincos_action

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """
        Convert the wrapped action, step the base environment, and add diagnostics.
        """

        phase_only_action = self.convert_sincos_to_phase_only(action)

        observation, reward, terminated, truncated, info = self.env.step(phase_only_action)

        if info is None:
            info = {}

        info = dict(info)

        info["wrapper_name"] = self.__class__.__name__
        info["wrapped_action_mode"] = "phase_sincos"
        info["base_action_mode"] = "phase_only"
        info["wrapped_action_dim"] = int(2 * self.num_elements)
        info["base_action_dim"] = int(self.num_elements)

        if self.store_last_converted_action:
            info["phase_only_action_from_wrapper"] = self.last_phase_only_action.copy()
            info["phase_rad_from_wrapper"] = self.last_phase_rad.copy()

        return observation, reward, terminated, truncated, info

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Reset the wrapped environment.
        """

        self.last_sincos_action = None
        self.last_phase_only_action = None
        self.last_phase_rad = None

        observation, info = self.env.reset(**kwargs)

        if info is None:
            info = {}

        info = dict(info)
        info["wrapper_name"] = self.__class__.__name__
        info["wrapped_action_mode"] = "phase_sincos"
        info["base_action_mode"] = "phase_only"
        info["wrapped_action_dim"] = int(2 * self.num_elements)
        info["base_action_dim"] = int(self.num_elements)

        return observation, info
    
class SupervisedActionRegularizationWrapper(gym.Wrapper):
    """
    Add supervised-action regularization to the training reward.

    This wrapper is intended for protected SAC fine-tuning from a supervised
    phase_sincos policy.

    The physical environment reward is preserved and a penalty is added:

        reward_total = reward_physical - lambda_reg * imitation_loss

    where imitation_loss is the mean squared error between the current agent
    action and the action predicted by a frozen supervised policy.

    This wrapper does not change the observation space or the action space.
    It only modifies the reward during training.

    Expected action format:
        action = [cos_1, ..., cos_N, sin_1, ..., sin_N]
    """

    def __init__(
        self,
        env: gym.Env,
        supervised_action_fn: Any,
        regularization_weight: float = 1.0,
        normalize_sincos: bool = True,
        eps: float = 1e-8,
        store_last_regularization_info: bool = True,
    ) -> None:
        super().__init__(env)

        self.supervised_action_fn = supervised_action_fn
        self.regularization_weight = float(regularization_weight)
        self.normalize_sincos = bool(normalize_sincos)
        self.eps = float(eps)
        self.store_last_regularization_info = bool(store_last_regularization_info)

        if not isinstance(self.action_space, spaces.Box):
            raise TypeError(
                "SupervisedActionRegularizationWrapper expects a Box action space."
            )

        if len(self.action_space.shape) != 1:
            raise ValueError(
                "SupervisedActionRegularizationWrapper expects a 1D action space. "
                f"Received shape: {self.action_space.shape}"
            )

        if self.action_space.shape[0] % 2 != 0:
            raise ValueError(
                "SupervisedActionRegularizationWrapper expects an even action dimension "
                "for phase_sincos actions."
            )

        self.num_elements = int(self.action_space.shape[0] // 2)

        self.current_observation: np.ndarray | None = None

        self.last_supervised_action: np.ndarray | None = None
        self.last_agent_action: np.ndarray | None = None
        self.last_imitation_loss: float | None = None
        self.last_supervised_action_penalty: float | None = None
        self.last_physical_reward: float | None = None
        self.last_regularized_reward: float | None = None

    def _normalize_sincos_action(self, action: np.ndarray) -> np.ndarray:
        """
        Normalize each (cos, sin) pair of a phase_sincos action.

        Input format:
            [cos_1, ..., cos_N, sin_1, ..., sin_N]

        Output format:
            [cos_1, ..., cos_N, sin_1, ..., sin_N]
        """

        action = np.asarray(action, dtype=np.float32).reshape(-1)

        expected_dim = 2 * self.num_elements

        if action.shape[0] != expected_dim:
            raise ValueError(
                f"Invalid phase_sincos action dimension. Expected {expected_dim}, "
                f"received {action.shape[0]}."
            )

        cos_values = action[: self.num_elements].astype(np.float64)
        sin_values = action[self.num_elements :].astype(np.float64)

        norm = np.sqrt(cos_values**2 + sin_values**2)
        norm = np.maximum(norm, self.eps)

        cos_values = cos_values / norm
        sin_values = sin_values / norm

        normalized_action = np.concatenate([cos_values, sin_values], axis=0)
        normalized_action = np.clip(normalized_action, -1.0, 1.0).astype(np.float32)

        return normalized_action

    def _compute_imitation_loss(
        self,
        agent_action: np.ndarray,
        supervised_action: np.ndarray,
    ) -> float:
        """
        Compute MSE between agent action and supervised action.
        """

        agent_action = np.asarray(agent_action, dtype=np.float32).reshape(-1)
        supervised_action = np.asarray(supervised_action, dtype=np.float32).reshape(-1)

        if agent_action.shape != supervised_action.shape:
            raise ValueError(
                "Agent action and supervised action must have the same shape. "
                f"Received {agent_action.shape} and {supervised_action.shape}."
            )

        if self.normalize_sincos:
            agent_action = self._normalize_sincos_action(agent_action)
            supervised_action = self._normalize_sincos_action(supervised_action)

        imitation_loss = float(
            np.mean(
                np.square(agent_action - supervised_action)
            )
        )

        return imitation_loss

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Reset the environment and store the initial observation.
        """

        observation, info = self.env.reset(**kwargs)

        self.current_observation = np.asarray(observation, dtype=np.float32).copy()

        self.last_supervised_action = None
        self.last_agent_action = None
        self.last_imitation_loss = None
        self.last_supervised_action_penalty = None
        self.last_physical_reward = None
        self.last_regularized_reward = None

        if info is None:
            info = {}

        info = dict(info)
        info["supervised_regularization_enabled"] = True
        info["supervised_regularization_weight"] = self.regularization_weight
        info["supervised_regularization_normalize_sincos"] = self.normalize_sincos

        return observation, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """
        Step the environment and regularize the reward.
        """

        action = np.asarray(action, dtype=np.float32).reshape(-1)

        if self.current_observation is None:
            raise RuntimeError(
                "Current observation is None. Call reset() before step()."
            )

        supervised_action = self.supervised_action_fn(self.current_observation)
        supervised_action = np.asarray(supervised_action, dtype=np.float32).reshape(-1)

        imitation_loss = self._compute_imitation_loss(
            agent_action=action,
            supervised_action=supervised_action,
        )

        observation, reward, terminated, truncated, info = self.env.step(action)

        physical_reward = float(reward)
        supervised_action_penalty = self.regularization_weight * imitation_loss
        regularized_reward = physical_reward - supervised_action_penalty

        if info is None:
            info = {}

        info = dict(info)

        info["physical_reward"] = physical_reward
        info["supervised_action_imitation_loss"] = imitation_loss
        info["supervised_action_penalty"] = supervised_action_penalty
        info["regularized_reward"] = regularized_reward
        info["supervised_regularization_enabled"] = True
        info["supervised_regularization_weight"] = self.regularization_weight
        info["supervised_regularization_normalize_sincos"] = self.normalize_sincos

        if self.store_last_regularization_info:
            self.last_supervised_action = supervised_action.copy()
            self.last_agent_action = action.copy()
            self.last_imitation_loss = imitation_loss
            self.last_supervised_action_penalty = supervised_action_penalty
            self.last_physical_reward = physical_reward
            self.last_regularized_reward = regularized_reward

        if terminated or truncated:
            self.current_observation = None
        else:
            self.current_observation = np.asarray(observation, dtype=np.float32).copy()

        return observation, regularized_reward, terminated, truncated, info