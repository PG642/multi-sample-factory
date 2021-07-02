"""
An example that shows how to use SampleFactory with a rocket_league_ env. To use this script, ypu need to add the rocket_league_saving_training_single executable in the root directory of this repo.

Example command line for saving_training_single:
python -m multi_sample_factory_examples.train_rocket_league_env --algo=APPO --use_rnn=False --num_envs_per_worker=20 --policy_workers_per_policy=2 --recurrence=1 --with_vtrace=False --batch_size=512 --hidden_size=256 --encoder_type=mlp --encoder_subtype=mlp_mujoco --reward_scale=0.1 --save_every_sec=10 --experiment_summaries_interval=10 --experiment=example_rocket_league_saving_training_single --env=rocket_league_saving_training_single
python -m multi_sample_factory_examples.enjoy_rocket_league_env --algo=APPO --experiment=example_enjoy_rocket_league_saving_training_single --env=rocket_league_saving_training_single

"""

import sys
import fcntl
import os
import random


import gym

from multi_sample_factory.algorithms.utils.arguments import arg_parser, parse_args
from multi_sample_factory.envs.env_registry import global_env_registry
from multi_sample_factory.run_algorithm import run_algorithm
from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper


def custom_parse_args(argv=None, evaluation=False):
    """
    Parse default SampleFactory arguments and add user-defined arguments on top.
    Setting the evaluation flag to True adds additional CLI arguments for evaluating the policy (see the enjoy_ scripts)

    """
    parser = arg_parser(argv, evaluation=evaluation)

    # insert additional parameters here if needed

    cfg = parse_args(argv=argv, evaluation=evaluation, parser=parser)
    return cfg


def make_rocket_league_env_func(full_env_name, cfg=None, env_config=None):
    assert full_env_name.startswith('rocket_league_')
    rocket_league_env_name = full_env_name.split('rocket_league_')[1]
    
    rand = random.SystemRandom().randint(-2147483648, 2147483647)
    
    if env_config != None:
        unity_env = UnityEnvironment(file_name=full_env_name,
                                     side_channels=[],
                                     worker_id=env_config.env_id,
                                     seed=rand)
    
    #this is a temporary environment with no env_config
    else:
        unity_env = UnityEnvironment(file_name=full_env_name,
                                     side_channels=[],
                                     worker_id=0,
                                     seed=rand)

    env = UnityToGymWrapper(unity_env)
    return env


def add_extra_params_func(env, parser):
    """Specify any additional command line arguments for this family of custom environments."""
    pass


def override_default_params_func(env, parser):
    """Override default argument values for this family of environments."""
    pass


def register_custom_components():
    global_env_registry().register_env(
        env_name_prefix='rocket_league_',
        make_env_func=make_rocket_league_env_func,
        add_extra_params_func=add_extra_params_func,
        override_default_params_func=override_default_params_func,
    )


def main():
    """Script entry point."""
    register_custom_components()
    cfg = custom_parse_args()
    status = run_algorithm(cfg)
    return status


if __name__ == '__main__':
    sys.exit(main())
