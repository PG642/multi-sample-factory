import math
from abc import ABC
from numbers import Number

import torch
from torch.distributions import Distribution, constraints
from torch.distributions.utils import broadcast_all


class TanhNormal(torch.distributions.Normal, ABC):
    def __init__(self, loc, scale):
        super().__init__(loc, scale)
        self.transform = torch.distributions.transforms.TanhTransform(cache_size=1)

    def sample(self, sample_shape=torch.Size([])):
        unsquashed_sample = super().sample(sample_shape)
        squashed = self.transform(unsquashed_sample)
        return squashed

    def log_prob(self, value):
        EPSILON = 1e-5  # Small value to avoid divide by zero
        capped_value = torch.clamp(value, -1 + EPSILON, 1 - EPSILON)
        unsquashed = self.transform.inv(capped_value)
        log_prob_of_normal_with_epsilon = self.log_prob_of_normal_with_epsilon(unsquashed)
        log_abs_det_jacobian = self.transform.log_abs_det_jacobian(unsquashed, capped_value)
        return log_prob_of_normal_with_epsilon - log_abs_det_jacobian

    def log_prob_of_normal_with_epsilon(self, value):
        """
        Log probability of the underlying normal distribution with epsilon to prevent nans.
        """
        EPSILON = 1e-7  # Small value to avoid divide by zero
        var = self.scale ** 2
        log_scale = torch.log(self.scale + EPSILON)
        return (
                -((value - self.loc) ** 2) / (2 * var + EPSILON)
                - log_scale
                - math.log(math.sqrt(2 * math.pi))
        )
