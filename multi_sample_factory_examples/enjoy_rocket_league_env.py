"""
An example that shows how to use SampleFactory with a Rocket League env.

Example command line for rocket_league_saving_trainig_single:
python -m multi_sample_factory_examples.enjoy_rocket_league_env --algo=APPO --experiment=example_rocket_league_saving_trainig_single --env=rocket_league_saving_trainig_single

"""

import sys

from mlagents_envs.environment import UnityEnvironment

from multi_sample_factory.algorithms.appo.enjoy_appo import enjoy
from multi_sample_factory.algorithms.utils.arguments import arg_parser, parse_args
from multi_sample_factory.envs.env_registry import global_env_registry


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

    if env_config != None:
        env = UnityEnvironment(file_name=full_env_name,
                                     side_channels=[],
                                     worker_id=env_config.env_id)

    # this is a temporary environment with no env_config
    else:
        env = UnityEnvironment(file_name=full_env_name,
                                     side_channels=[],
                                     worker_id=0)
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
    # This is a non-blocking call that only loads the environment.
    env = UnityEnvironment(file_name="rocket_league_saving_trainig_single", seed=1, side_channels=[])
    # Start interacting with the environment.
    env.reset()
    behavior_names = env.behavior_specs.keys()


if __name__ == '__main__':
    sys.exit(main())