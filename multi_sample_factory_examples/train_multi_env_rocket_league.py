"""
From the root of Sample Factory repo this can be run as:
python -m multi_sample_factory_examples.train_custom_multi_env --algo=APPO --env=my_custom_multi_env_v1 --experiment=example --save_every_sec=5 --experiment_summaries_interval=10

After training for a desired period of time, evaluate the policy by running:
python -m multi_sample_factory_examples.enjoy_custom_multi_env --algo=APPO --env=my_custom_multi_env_v1 --experiment=example

"""
import random
import sys

import gym
import numpy as np

from mlagents_envs.environment import UnityEnvironment
from multi_sample_factory_examples.multi_env_gym_wrapper import UnityToGymWrapper
from multi_sample_factory.algorithms.utils.arguments import arg_parser, parse_args
from multi_sample_factory.envs.env_registry import global_env_registry
from multi_sample_factory.run_algorithm import run_algorithm

class MultiEnvRocket(gym.Env):
    """
    Implements a simple 2-agent game. Observation space is irrelevant. Optimal strategy is for both agents
    to choose the same action (both 0 or 1).

    """
    def __init__(self, full_env_name, cfg):
        self.name = full_env_name  # optional
        self.cfg = cfg

        file_path = "/work/grudelpg/executables/multi_agent_env/StandaloneLinux64"
        env = UnityEnvironment(file_path)

        self.env = UnityToGymWrapper(env)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        print(self.observation_space)
        print(type(self.observation_space))
        print(self.action_space)
        print(type(self.action_space))

        self.num_agents = self.env.num_agents
        self.is_multiagent = True 

    def reset(self):
        return self.env.reset()

    def step(self, actions):
        info = [dict() for _ in range(self.num_agents)]
        obs_n, reward_n, done_n, _ =  self.env.step(actions)

        # check if done
        if all(done_n) == True:
            self.reset()

        return obs_n, reward_n, done_n, info

    def render(self, mode='human'):
        pass

def make_multi_agent_rocket(full_env_name, cfg=None, env_config=None):
    return MultiEnvRocket(full_env_name, cfg)


def register_custom_components():
    global_env_registry().register_env(
        env_name_prefix='rocket_league_ma',
        make_env_func=make_multi_agent_rocket
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