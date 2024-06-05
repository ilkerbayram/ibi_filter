#!/usr/bin/env python
"""
this module implements the estimator in 'Robust Gaussian Kalman
Filter with Outlier Detection', by Wang et al., IEEE SPL, 2018
for a simple special case, specifically we take 
1. the state update to be a random walk, 
2. measurements to be noisy observations of the state

Ilker Bayram, ibayram@ieee.org, 2024
"""

import numpy as np
import numpy.typing as npt
from scipy.special import digamma
from typing import Tuple


class RobustGaussian:
    """
    This class implements the estimator
    for the following state update equations
    for f(x) = x, and h(x) = x in the paper

    Further, we take x to be a scalar
    """

    def __init__(
        self,
        initial_mean: float,
        initial_variance: float,
        state_update_noise_variance: float,
        observation_noise_variance: float,
        e_0: float = 0.9,
        f_0: float = 0.1,
    ) -> "RobustGaussian":
        self.mean = initial_mean
        self.variance = initial_variance
        self.state_update_noise_variance = state_update_noise_variance
        self.observation_noise_variance = observation_noise_variance
        self.e_0 = e_0
        self.f_0 = f_0
        # the following are placeholders,
        # not really needed at initialization
        self.z_avg = 1
        self.x_hat = initial_mean
        self.p_hat = initial_variance
        self.z_avg_threshold = 1e-10  # threshold to decide if outlier
        self.e_t = e_0
        self.f_t = f_0
        self.max_inner_iter = 5

    def propagate_mean(self) -> None:
        self.x_hat = self.mean

    def propagate_variance(self) -> None:
        self.p_hat = self.variance + self.state_update_noise_variance

    def propagate(self) -> None:
        self.propagate_mean()
        self.propagate_variance()

    def update_qx(self, new_observation: float) -> None:
        # treat observation as outlier if z is low enough
        if self.z_avg < self.z_avg_threshold:
            self.mean = self.x_hat
            self.variance = self.p_hat
            return None

        # the integrals evaluate to simple expressions
        # for this case
        y_hat = self.x_hat
        S_t = self.p_hat
        C_t = self.p_hat
        U_t = 1 / (S_t + self.observation_noise_variance / self.z_avg)
        K_t = C_t * U_t
        self.mean = self.x_hat + K_t * (new_observation - y_hat)
        self.variance = self.p_hat - C_t * U_t * C_t
        return None

    def update_qz(self, new_observation: float) -> None:
        integral_term = (
            -0.5
            * (1 / self.observation_noise_variance)
            * (
                new_observation**2
                - 2 * new_observation * self.mean
                + self.variance
            )
        )
        prob_z_1 = np.exp(
            digamma(self.e_t) - digamma(self.e_t + self.f_t + 1) + integral_term
        )
        prob_z_0 = np.exp(
            digamma(self.f_t + 1) - digamma(self.e_t + self.f_t + 1)
        )
        self.z_avg = prob_z_1 / (prob_z_1 + prob_z_0)
        return None

    def update_qpi(self) -> None:
        self.e_t = self.e_0 + self.z_avg
        self.f_t = self.f_0 + 1 - self.z_avg
        return None

    def __call__(self, new_observation: float) -> Tuple[float]:
        """
        returns updated mean, variance, prob(outlier)
        """
        self.propagate()
        self.z_avg = 1
        for _ in range(self.max_inner_iter):
            self.update_qx(new_observation=new_observation)
            self.update_qz(new_observation=new_observation)
            self.update_qpi()

        return self.mean, self.variance, (1 - self.z_avg)
