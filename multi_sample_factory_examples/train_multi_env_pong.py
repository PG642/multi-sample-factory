"""
From the root of Sample Factory repo this can be run as:
python -m multi_sample_factory_examples.train_custom_multi_env --algo=APPO --env=my_custom_multi_env_v1 --experiment=example --save_every_sec=5 --experiment_summaries_interval=10

After training for a desired period of time, evaluate the policy by running:
python -m multi_sample_factory_examples.enjoy_custom_multi_env --algo=APPO --env=my_custom_multi_env_v1 --experiment=example

"""
import random
import sys

import gym
import ma_gym
import numpy as np

from multi_sample_factory.algorithms.utils.arguments import arg_parser, parse_args
from multi_sample_factory.envs.env_registry import global_env_registry
from multi_sample_factory.run_algorithm import run_algorithm

class MultiEnvPong(gym.Env):
    """
    Implements a simple 2-agent game. Observation space is irrelevant. Optimal strategy is for both agents
    to choose the same action (both 0 or 1).

    """
    def __init__(self, full_env_name, cfg):
        self.name = full_env_name  # optional
        self.cfg = cfg

        self.env = gym.make('ma_gym:PongDuel-v0')
        self.observation_space = self.env.observation_space[0]
        self.action_space = self.env.action_space[0]

        self.num_agents = self.env.n_agents
        self.is_multiagent = True 

    def reset(self):
        return [np.array(obs) for obs in self.env.reset()]

    def step(self, actions):
        info = [dict() for _ in range(self.num_agents)]
        obs_n, reward_n, done_n, _ =  self.env.step(actions)

        # check if done
        if all(done_n) == True:
            self.reset()

        return [np.array(obs) for obs in obs_n], reward_n, done_n, info

    def render(self, mode='human'):
        self.env.render(mode)

def make_multi_agent_pong(full_env_name, cfg=None, env_config=None):
    return MultiEnvPong(full_env_name, cfg)


def register_custom_components():
    global_env_registry().register_env(
        env_name_prefix='multi_agent_pong',
        make_env_func=make_multi_agent_pong
    )

def custom_parse_args(argv=None, evaluation=False):
    """
    Parse default SampleFactory arguments and add user-defined arguments on top.
    Allow to override argv for unit tests. Default value (None) means use sys.argv.
    Setting the evaluation flag to True adds additional CLI arguments for evaluating the policy (see the enjoy_ script).

    """
    parser = arg_parser(argv, evaluation=evaluation)
    parser.set_defaults(
        encoder_custom=None
    )
    # SampleFactory parse_args function does some additional processing (see comments there)
    cfg = parse_args(argv=argv, evaluation=evaluation, parser=parser)
    return cfg


def main():
    """Script entry point."""
    register_custom_components()
    cfg = custom_parse_args()
    
    status = run_algorithm(cfg)
    return status


if __name__ == '__main__':
    sys.exit(main())