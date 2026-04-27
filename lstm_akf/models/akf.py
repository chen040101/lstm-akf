from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np


@dataclass
class StateEstimatorConfig:
    dt: float = 1.0
    max_missing: int = 5
    process_noise: float = 1e-4
    measurement_noise: float = 1e-3
    initial_covariance: float = 1.0
    adaptive_measurement_noise: bool = True
    measurement_noise_min_scale: float = 0.25
    measurement_noise_max_scale: float = 16.0
    innovation_gain: float = 1.0
    innovation_smoothing: float = 0.2
    confidence_gain: float = 1.0
    missing_process_noise_scale: float = 1.35
    max_process_noise_scale: float = 6.0


class XYStateEstimator:
    def __init__(self, config: Optional[StateEstimatorConfig] = None) -> None:
        self.config = config or StateEstimatorConfig()
        self._transition = self._build_transition(self.config.dt)
        self._measurement = np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        self._base_process_noise = np.eye(6, dtype=np.float64) * float(self.config.process_noise)
        self._base_measurement_noise = np.eye(2, dtype=np.float64) * float(self.config.measurement_noise)
        self.reset()

    @staticmethod
    def _build_transition(dt: float) -> np.ndarray:
        return np.array(
            [
                [1.0, 0.0, dt, 0.0, 0.5 * dt * dt, 0.0],
                [0.0, 1.0, 0.0, dt, 0.0, 0.5 * dt * dt],
                [0.0, 0.0, 1.0, 0.0, dt, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    def reset(self) -> None:
        self._state: Optional[np.ndarray] = None
        self._covariance: Optional[np.ndarray] = None
        self._missing_count = 0
        self._measurement_noise_scale = 1.0

    def is_active(self) -> bool:
        return self._state is not None

    def current_state(self) -> Optional[np.ndarray]:
        if self._state is None:
            return None
        return self._state.copy()

    def current_xy(self) -> Optional[np.ndarray]:
        if self._state is None:
            return None
        return self._state[:2].copy()

    def _initialize(self, observation_xy: Sequence[float]) -> None:
        x, y = float(observation_xy[0]), float(observation_xy[1])
        self._state = np.array([x, y, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self._covariance = np.eye(6, dtype=np.float64) * float(self.config.initial_covariance)
        self._missing_count = 0
        self._measurement_noise_scale = 1.0

    def _process_noise_scale(self, missing_count: int) -> float:
        if missing_count <= 0:
            return 1.0
        scale = float(self.config.missing_process_noise_scale) ** float(missing_count)
        return float(np.clip(scale, 1.0, float(self.config.max_process_noise_scale)))

    def _predict_once(self, process_scale: float = 1.0) -> None:
        assert self._state is not None and self._covariance is not None
        process_noise = self._base_process_noise * float(process_scale)
        self._state = self._transition @ self._state
        self._covariance = self._transition @ self._covariance @ self._transition.T + process_noise

    def _adaptive_measurement_noise(
        self,
        innovation: np.ndarray,
        predicted_measurement_covariance: np.ndarray,
        confidence: Optional[float],
    ) -> np.ndarray:
        if not self.config.adaptive_measurement_noise:
            return self._base_measurement_noise.copy()

        expected_variance = float(
            np.trace(predicted_measurement_covariance) / predicted_measurement_covariance.shape[0]
        )
        expected_std = max(expected_variance, 1e-8) ** 0.5
        innovation_norm = float(np.linalg.norm(innovation))
        normalized_innovation = innovation_norm / max(expected_std, 1e-6)

        target_scale = 1.0 + (float(self.config.innovation_gain) * (normalized_innovation - 1.0))
        target_scale = float(
            np.clip(
                target_scale,
                float(self.config.measurement_noise_min_scale),
                float(self.config.measurement_noise_max_scale),
            )
        )

        if confidence is not None:
            confidence = float(np.clip(confidence, 0.0, 1.0))
            target_scale *= 1.0 + (float(self.config.confidence_gain) * (1.0 - confidence))

        target_scale = float(
            np.clip(
                target_scale,
                float(self.config.measurement_noise_min_scale),
                float(self.config.measurement_noise_max_scale),
            )
        )

        smoothing = float(np.clip(self.config.innovation_smoothing, 0.0, 1.0))
        self._measurement_noise_scale = ((1.0 - smoothing) * self._measurement_noise_scale) + (
            smoothing * target_scale
        )
        self._measurement_noise_scale = float(
            np.clip(
                self._measurement_noise_scale,
                float(self.config.measurement_noise_min_scale),
                float(self.config.measurement_noise_max_scale),
            )
        )
        return self._base_measurement_noise * self._measurement_noise_scale

    def update(
        self,
        observation_xy: Optional[Sequence[float]],
        confidence: Optional[float] = None,
    ) -> Dict[str, object]:
        if observation_xy is None:
            if self._state is None:
                return {
                    "active": False,
                    "xy": None,
                    "missing_count": 0,
                    "observed": False,
                    "process_noise_scale": 1.0,
                    "measurement_noise_scale": self._measurement_noise_scale,
                }

            next_missing_count = self._missing_count + 1
            process_scale = self._process_noise_scale(next_missing_count)
            self._predict_once(process_scale=process_scale)
            self._missing_count = next_missing_count
            if self._missing_count > self.config.max_missing:
                self.reset()
                return {
                    "active": False,
                    "xy": None,
                    "missing_count": self.config.max_missing + 1,
                    "observed": False,
                    "process_noise_scale": process_scale,
                    "measurement_noise_scale": self._measurement_noise_scale,
                }
            return {
                "active": True,
                "xy": self.current_xy(),
                "missing_count": self._missing_count,
                "observed": False,
                "process_noise_scale": process_scale,
                "measurement_noise_scale": self._measurement_noise_scale,
            }

        measurement = np.asarray(observation_xy, dtype=np.float64)
        if self._state is None:
            self._initialize(measurement)
            return {
                "active": True,
                "xy": self.current_xy(),
                "missing_count": 0,
                "observed": True,
                "process_noise_scale": 1.0,
                "measurement_noise_scale": self._measurement_noise_scale,
            }

        process_scale = self._process_noise_scale(self._missing_count)
        self._predict_once(process_scale=process_scale)
        innovation = measurement - (self._measurement @ self._state)
        predicted_measurement_covariance = self._measurement @ self._covariance @ self._measurement.T
        measurement_noise = self._adaptive_measurement_noise(
            innovation,
            predicted_measurement_covariance,
            confidence,
        )
        innovation_covariance = predicted_measurement_covariance + measurement_noise
        kalman_gain = self._covariance @ self._measurement.T @ np.linalg.inv(innovation_covariance)
        self._state = self._state + (kalman_gain @ innovation)
        identity = np.eye(self._covariance.shape[0], dtype=np.float64)
        self._covariance = (identity - (kalman_gain @ self._measurement)) @ self._covariance
        self._missing_count = 0
        return {
            "active": True,
            "xy": self.current_xy(),
            "missing_count": 0,
            "observed": True,
            "process_noise_scale": process_scale,
            "measurement_noise_scale": self._measurement_noise_scale,
        }

    def predict_future(self, future_steps: int) -> np.ndarray:
        if future_steps <= 0:
            return np.zeros((0, 2), dtype=np.float32)
        if self._state is None or self._covariance is None:
            return np.zeros((future_steps, 2), dtype=np.float32)

        state = self._state.copy()
        covariance = self._covariance.copy()
        predictions = np.zeros((future_steps, 2), dtype=np.float64)
        for index in range(future_steps):
            state = self._transition @ state
            covariance = self._transition @ covariance @ self._transition.T + self._base_process_noise
            predictions[index] = state[:2]
        return predictions.astype(np.float32)


__all__ = ["StateEstimatorConfig", "XYStateEstimator"]
