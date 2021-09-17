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

    def _inverse_tanh(self, value):
        EPSILON = 1e-7  # Small value to avoid divide by zero

        capped_value = torch.clamp(value, -1 + EPSILON, 1 - EPSILON)
        return 0.5 * torch.log((1 + capped_value) / (1 - capped_value) + EPSILON)

    def log_prob(self, value):
        unsquashed = self.transform.inv(value)
        return self.log_prob_of_normal_with_epsilon(unsquashed) - self.transform.log_abs_det_jacobian(
            unsquashed, value
        )

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
