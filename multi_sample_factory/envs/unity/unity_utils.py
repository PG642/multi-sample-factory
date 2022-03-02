from random import random
from typing import List

from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel import  EnvironmentParametersChannel
from os.path import join
import random
from multi_sample_factory.envs.unity.UnityGymWrapper import UnityToGymWrapper as MultiUnityToGymWrapper



class UnitySpec:
    def __init__(self, full_env_name, exec_file_name, env_parameters: List[str]):
        self.full_env_name = full_env_name
        self.exec_file_name = exec_file_name
        self.env_parameters= env_parameters


UNITY_ENVS = [
    UnitySpec('unity_saving_training_continuous', 'saving_training_continuous', []),
    UnitySpec('unity_saving_training_discrete',
              'saving_training_discrete',
              ["seed", "difficulty", "initialBoost", "canDoubleJump", "canDrift", "canBoost", "useSuspension", "useBulletImpulse",
               "usePsyonixImpulse", "useCustomBounce", "useWallStabilization", "useGroundStabilization"]),
    UnitySpec('unity_saving_training_mixed', 'saving_training_mixed', []),
    UnitySpec('unity_rocket_league_saving_training_single_discrete', 'rocket_league_saving_training_single_discrete', []),
    UnitySpec('unity_striker', 'striker', []),
    # You can add more unity environments here if needed
]

UNITY_MULTI_ENVS = [
    UnitySpec('unity_multi_agent_env', 'multi_agent_env', []),
    UnitySpec('unity_1v1', "1v1", []),
    UnitySpec("unity_1v1alternativeVerteiltesLernen", "1v1alternativeVerteiltesLernen", []),
]


def unity_env_by_name(full_env_name):
    for cfg in UNITY_ENVS:
        if cfg.full_env_name == full_env_name:
            return cfg
    for cfg in UNITY_MULTI_ENVS:
        if cfg.full_env_name == full_env_name:
            return cfg
    raise Exception('Unknown Unity env')


def make_unity_env(full_env_name, cfg, env_config=None):
    unity_spec = unity_env_by_name(full_env_name)
    rand = random.SystemRandom().randint(-2147483648, 2147483647)
    exec_path = join(cfg.exec_dir, unity_spec.exec_file_name, unity_spec.exec_file_name)
    engineConfigChannel = EngineConfigurationChannel()
    env_parameter_channel = EnvironmentParametersChannel()

    if env_config is not None:
        unity_env = UnityEnvironment(file_name=exec_path,
                                    side_channels=[engineConfigChannel, env_parameter_channel],
                                    worker_id=env_config.env_id,
                                    seed=rand)

    # this is a temporary environment with no env_config
    else:
        unity_env = UnityEnvironment(file_name=exec_path,
                                    side_channels=[engineConfigChannel],
                                    worker_id=0,
                                    seed=rand)
    engineConfigChannel.set_configuration_parameters(time_scale=cfg.unity_time_scale)
    for key, value in cfg.env_params.items():
        if key in unity_spec.env_parameters:
            env_parameter_channel.set_float_parameter(key=key, value=value)
        else:
            raise ValueError("Unknown environment parameter {0} detected. Available environment parameters for "
                            "environment {1} are {2}".format(key, full_env_name, unity_spec.env_parameters))
    # env = UnityToGymWrapper(unity_env)
    # return env
# check if the environment is for a single agent
    if unity_spec in [spec.full_env_name for spec in UNITY_ENVS]:
        env = UnityToGymWrapper(unity_env)
        return env
    elif unity_spec in [spec.full_env_name for spec in UNITY_MULTI_ENVS]:
        env = MultiUnityToGymWrapper(unity_env)
        return env
    else
        raise ValueError("Environment Spec is not properly used.")