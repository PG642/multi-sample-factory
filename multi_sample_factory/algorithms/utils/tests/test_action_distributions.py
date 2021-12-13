import random
import time
from unittest import TestCase

import gym
import numpy as np
import torch
from gym.spaces import MultiDiscrete, Discrete, Tuple,Box, MultiBinary
from torch.distributions import Categorical

from multi_sample_factory.algorithms.utils.action_distributions import get_action_distribution, calc_num_logits, \
    sample_actions_log_probs, transform_action_space, calc_num_actions
from multi_sample_factory.utils.timing import Timing
from multi_sample_factory.utils.utils import log


class TestActionDistributions(TestCase):
    batch_size = 4  # whatever

    def test_simple_distribution(self):
        simple_action_space = gym.spaces.Discrete(3)
        simple_num_logits = calc_num_logits(simple_action_space)
        self.assertEqual(simple_num_logits, simple_action_space.n)

        simple_logits = torch.rand(self.batch_size, simple_num_logits)
        simple_action_distribution = get_action_distribution(simple_action_space, simple_logits)

        simple_actions = simple_action_distribution.sample()
        self.assertEqual(list(simple_actions.shape), [self.batch_size])
        self.assertTrue(all(0 <= a < simple_action_space.n for a in simple_actions))

    def test_gumbel_trick(self):
        """
        We use a Gumbel noise which seems to be faster compared to using pytorch multinomial.
        Here we test that those are actually equivalent.
        """

        timing = Timing()

        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        with torch.no_grad():
            action_space = gym.spaces.Discrete(8)
            num_logits = calc_num_logits(action_space)
            device_type = 'cpu'
            device = torch.device(device_type)
            logits = torch.rand(self.batch_size, num_logits, device=device) * 10.0 - 5.0

            if device_type == 'cuda':
                torch.cuda.synchronize(device)

            count_gumbel, count_multinomial = np.zeros([action_space.n]), np.zeros([action_space.n])

            # estimate probability mass by actually sampling both ways
            num_samples = 20000

            action_distribution = get_action_distribution(action_space, logits)
            sample_actions_log_probs(action_distribution)
            action_distribution.sample_gumbel()

            with timing.add_time('gumbel'):
                for i in range(num_samples):
                    action_distribution = get_action_distribution(action_space, logits)
                    samples_gumbel = action_distribution.sample_gumbel()
                    count_gumbel[samples_gumbel[0]] += 1

            action_distribution = get_action_distribution(action_space, logits)
            action_distribution.sample()

            with timing.add_time('multinomial'):
                for i in range(num_samples):
                    action_distribution = get_action_distribution(action_space, logits)
                    samples_multinomial = action_distribution.sample()
                    count_multinomial[samples_multinomial[0]] += 1

            estimated_probs_gumbel = count_gumbel / float(num_samples)
            estimated_probs_multinomial = count_multinomial / float(num_samples)

            log.debug('Gumbel estimated probs: %r', estimated_probs_gumbel)
            log.debug('Multinomial estimated probs: %r', estimated_probs_multinomial)
            log.debug('Sampling timing: %s', timing)
            time.sleep(0.1)  # to finish logging


    def test_sanity(self):
        raw_logits = torch.tensor([[0.0, 1.0, 2.0]])
        action_space = gym.spaces.Discrete(3)
        categorical = get_action_distribution(action_space, raw_logits)

        torch_categorical = Categorical(logits=raw_logits)
        torch_categorical_log_probs = torch_categorical.log_prob(torch.tensor([0, 1, 2]))

        entropy = categorical.entropy()
        torch_entropy = torch_categorical.entropy()
        self.assertTrue(np.allclose(entropy.numpy(), torch_entropy))

        log_probs = [categorical.log_prob(torch.tensor([action])) for action in [0, 1, 2]]
        log_probs = torch.cat(log_probs)

        self.assertTrue(np.allclose(torch_categorical_log_probs.numpy(), log_probs.numpy()))

        probs = torch.exp(log_probs)

        expected_probs = np.array([0.09003057317038046, 0.24472847105479764, 0.6652409557748219])

        self.assertTrue(np.allclose(probs.numpy(), expected_probs))

        tuple_space = gym.spaces.Tuple([action_space, action_space])
        raw_logits = torch.tensor([[0.0, 1.0, 2.0, 0.0, 1.0, 2.0]])
        tuple_distr = get_action_distribution(tuple_space, raw_logits)

        for a1 in [0, 1, 2]:
            for a2 in [0, 1, 2]:
                action = torch.tensor([[a1, a2]])
                log_prob = tuple_distr.log_prob(action)
                probability = torch.exp(log_prob)[0].item()
                self.assertAlmostEqual(probability, expected_probs[a1] * expected_probs[a2], delta=1e-6)

    def test_calc_num_actions(self):
        # Discrete action space
        action_space = Discrete(2)
        num_actions = calc_num_actions(action_space)
        self.assertEqual(num_actions, 1)

        # Continuous action space
        action_space = Box(high=1, low=-1, shape=[3], dtype=np.float32)
        num_actions = calc_num_actions(action_space)
        self.assertEqual(num_actions, 3)

        # Tuple action space
        discrete_action_spaces = [Discrete(3), Discrete(2), Discrete(4)]
        continuous_action_space = Box(high=1, low=-1, shape=[3], dtype=np.float32)
        sub_spaces = [continuous_action_space]
        sub_spaces.extend(discrete_action_spaces)
        action_space = Tuple(sub_spaces)
        num_actions = calc_num_actions(action_space)
        self.assertEqual(num_actions, 6)

    def test_calc_num_logits(self):
        # Test with discrete action space
        action_space = Discrete(2)
        num_logits = calc_num_logits(action_space)
        self.assertEqual(num_logits, 2)

        # Test with tuple action space
        action_space = Tuple((Discrete(2), Discrete(3)))
        num_logits = calc_num_logits(action_space)
        self.assertEqual(num_logits, 5)

        # Test with box action space in R^1
        action_space = Box(low=-1.0, high=2.0, shape=(3,), dtype=np.float64)
        num_logits = calc_num_logits(action_space)
        self.assertEqual(num_logits, 6)

        # Test with box action space in R^2
        action_space = Box(low=-1.0, high=2.0, shape=(3, 4), dtype=np.float64)
        num_logits = calc_num_logits(action_space)
        self.assertEqual(num_logits, 24)

        # Test with multi discrete action space
        action_space= MultiDiscrete([5, 5, 5, 2])
        action_space = transform_action_space(action_space)
        self.assertIsInstance(action_space, Tuple)
        num_logits = calc_num_logits(action_space)
        self.assertEqual(num_logits, 17)

        # Test with mixed action space
        continuous_space = Box(low=-1.0, high=2.0, shape=(3,), dtype=np.float64)
        multi_discrete_space = MultiDiscrete([5, 5, 5, 2])
        action_space = Tuple((continuous_space,multi_discrete_space))
        action_space = transform_action_space((action_space))
        self.assertIsInstance(action_space, Tuple)
        num_logits = calc_num_logits(action_space)
        self.assertEqual(num_logits, 23)
