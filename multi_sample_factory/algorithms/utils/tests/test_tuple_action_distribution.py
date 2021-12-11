import random
from unittest import TestCase

import gym
import numpy as np
import torch
from gym.spaces import MultiDiscrete, Discrete, Tuple, Box

from multi_sample_factory.algorithms.utils.action_distributions import get_action_distribution, calc_num_logits, \
    transform_action_space, ContinuousActionDistribution


class TestTupleActionDistribution(TestCase):
    """
    Unittests for the tuple action distribution class. Run all test cases in this class with:
    python -m unittest multi_sample_factory.algorithms.utils.tests.test_tuple_action_distribution.TestTupleActionDistribution
    """

    def setUp(self) -> None:
        """
        Sets the batch size.
        Returns
        -------
        None
            None
        """
        self.batch_size = 5

    def test_tuple_distribution(self):
        num_spaces = random.randint(1, 4)
        spaces = [Discrete(random.randint(2, 5)) for _ in range(num_spaces)]
        action_space = gym.spaces.Tuple(spaces)

        num_logits = calc_num_logits(action_space)
        logits = torch.rand(self.batch_size, num_logits)
        self.assertEqual(num_logits, sum(s.n for s in action_space.spaces))

        action_distribution = get_action_distribution(action_space, logits)

        tuple_actions = action_distribution.sample()
        self.assertEqual(list(tuple_actions.shape), [self.batch_size, num_spaces])

        log_probs = action_distribution.log_prob(tuple_actions)
        self.assertEqual(list(log_probs.shape), [self.batch_size])

        entropy = action_distribution.entropy()
        self.assertEqual(list(entropy.shape), [self.batch_size])

    def test_tuple_distribution_with_mixed_space(self):
        # Run with: python -m unittest multi_sample_factory.algorithms.utils.tests.test_action_distributions.TestActionDistributions.test_tuple_distribution_with_mixed_space
        continuous_space = Box(low=-1.0, high=1.0, shape=[3], dtype=np.float32)
        multi_discrete_space = MultiDiscrete([3, 2, 2, 2, 2])
        num_spaces = continuous_space.shape[0] + len(multi_discrete_space.nvec)
        action_space = Tuple([continuous_space, multi_discrete_space])
        action_space = transform_action_space(action_space)
        num_logits = calc_num_logits(action_space)
        logits = torch.rand(self.batch_size, num_logits)
        action_distribution = get_action_distribution(action_space, logits)

        # Sampling actions and log probs at the same time
        batch_of_action_tuples, log_probs = action_distribution.sample_actions_log_probs()
        self.assertEqual(list(batch_of_action_tuples.shape), [self.batch_size, num_spaces])
        self.assertEqual(list(log_probs.shape), [self.batch_size])

        # Sampling actions
        tuple_actions = action_distribution.sample()
        self.assertEqual(list(tuple_actions.shape), [self.batch_size, num_spaces])

        # Log probs
        log_probs = action_distribution.log_prob(tuple_actions)
        self.assertEqual(list(log_probs.shape), [self.batch_size])

        # Entropy
        entropy = action_distribution.entropy()
        self.assertEqual(list(entropy.shape), [self.batch_size])

        #KL Divergence
        other_logits = torch.rand(self.batch_size, num_logits)
        other_action_distribution = get_action_distribution(action_space, other_logits)
        kl_divergence = action_distribution.kl_divergence(other_action_distribution)
        self.assertEqual(list(kl_divergence.shape), [self.batch_size])


    def test_tuple_sanity_check(self):
        num_spaces, num_actions = 3, 2
        simple_space = gym.spaces.Discrete(num_actions)
        spaces = [simple_space for _ in range(num_spaces)]
        tuple_space = gym.spaces.Tuple(spaces)

        self.assertTrue(calc_num_logits(tuple_space), num_spaces * num_actions)

        simple_logits = torch.zeros(1, num_actions)
        tuple_logits = torch.zeros(1, calc_num_logits(tuple_space))

        simple_distr = get_action_distribution(simple_space, simple_logits)
        tuple_distr = get_action_distribution(tuple_space, tuple_logits)

        tuple_entropy = tuple_distr.entropy()
        self.assertEqual(tuple_entropy, simple_distr.entropy() * num_spaces)

        simple_logprob = simple_distr.log_prob(torch.ones(1))
        tuple_logprob = tuple_distr.log_prob(torch.ones(1, num_spaces))
        self.assertEqual(tuple_logprob, simple_logprob * num_spaces)
