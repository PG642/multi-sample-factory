from unittest import TestCase

import torch
from gym.spaces import Tuple, Box
import numpy as np

from multi_sample_factory.algorithms.utils.action_distributions import calc_num_logits, get_action_distribution


class TestContinuousActionDistribution(TestCase):

    def setUp(self) -> None:
        self.batch_size = 5

    def test_continuous_distribution(self):
        num_actions = 3
        action_space = Box(high=-1, low=-1, shape=[num_actions], dtype=np.float32)

        num_logits = calc_num_logits(action_space)
        logits = torch.rand(self.batch_size, num_logits)
        self.assertEqual(num_logits, num_actions*2)

        action_distribution = get_action_distribution(action_space, logits)

        tuple_actions = action_distribution.sample()
        self.assertEqual(list(tuple_actions.shape), [self.batch_size, num_actions])

        log_probs = action_distribution.log_prob(tuple_actions)
        self.assertEqual(list(log_probs.shape), [self.batch_size])

        entropy = action_distribution.entropy()
        self.assertEqual(list(entropy.shape), [self.batch_size])