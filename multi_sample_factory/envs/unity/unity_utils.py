from random import random
from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from os.path import join
import random


class UnitySpec:
    def __init__(self, full_env_name, exec_file_name):
        self.full_env_name = full_env_name
        self.exec_file_name = exec_file_name


UNITY_ENVS = [
    UnitySpec('unity_rocket_league_saving_training_single', 'rocket_league_saving_training_single'),
    UnitySpec('unity_rocket_league_saving_training_single_discrete', 'rocket_league_saving_training_single_discrete')
    # You can add more unity environments here if needed.
]


def unity_env_by_name(full_env_name):
    for cfg in UNITY_ENVS:
        if cfg.full_env_name == full_env_name:
            return cfg
    raise Exception('Unknown Unity env')


def make_unity_env(full_env_name, cfg, env_config=None):
    unity_spec = unity_env_by_name(full_env_name)
    rand = random.SystemRandom().randint(-2147483648, 2147483647)
    exec_path = join(cfg.exec_dir, unity_spec.exec_file_name, unity_spec.exec_file_name)
    engineConfigChannel = EngineConfigurationChannel()
    if env_config is not None:
        unity_env = UnityEnvironment(file_name=exec_path,
                                     side_channels=[engineConfigChannel],
                                     worker_id=env_config.env_id,
                                     seed=rand)

    # this is a temporary environment with no env_config
    else:
        unity_env = UnityEnvironment(file_name=exec_path,
                                     side_channels=[engineConfigChannel],
                                     worker_id=0,
                                     seed=rand)
    engineConfigChannel.set_configuration_parameters(time_scale=cfg.unity_time_scale)
    env = UnityToGymWrapper(unity_env)
    return env
