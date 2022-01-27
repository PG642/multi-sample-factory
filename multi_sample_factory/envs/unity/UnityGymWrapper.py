import numpy as np
from typing import Any, List, Optional, Tuple, Union

import gym
from gym import error, spaces, Space

from mlagents_envs.base_env import ActionTuple, BaseEnv
from mlagents_envs.base_env import DecisionSteps, TerminalSteps
from mlagents_envs import logging_util
from gym import spaces

from itertools import chain


class UnityGymException(error.Error):
    """
    Any error related to the gym wrapper of ml-agents.
    """
    pass


logger = logging_util.get_logger(__name__)
logging_util.set_log_level(logging_util.INFO)


class UnityToGymWrapper(gym.Env):
    """
    Provides Gym wrapper for Unity Learning Environments.
    """

    def __init__(
            self,
            unity_env: BaseEnv,
            action_space_seed: Optional[int] = None,
    ):
        """
        Initialize the Wrapper by saving the existing behaviours and creating corresponding action
        and observation spaces
        :param unity_env: The Unity BaseEnv to be wrapped in the gym. Will be closed when the UnityToGymWrapper closes.
        :param action_space_seed: If non-None, will be used to set the random seed on created gym.Space instances.
        """
        self._env = unity_env

        # Take a single step so that the brain information will be sent over
        if not self._env.behavior_specs:
            self._env.step()

        # Hidden flag used by Atari environments to determine if the game is over
        self.game_over = False

        # Each agent can have a different behaviour. We want to be able to access each behaviour later on.
        self.behaviour_names = list(self._env.behavior_specs.keys())
        self.behaviour_specs = self._env.behavior_specs

        # Save the step result from the last time all Agents requested decisions.
        self.num_agents = 0
        self._previous_decision_steps = {}
        print("Reset", self._env.reset(), flush = True)
        for name in self.behaviour_names:
            decision_steps, _ = self._env.get_steps(name)
            self._previous_decision_steps[name] = decision_steps
            self.num_agents += len(decision_steps)

        # Set action spaces for each behaviour
        self._action_space = {}

        for name in self.behaviour_names:
            action_spec = self.behaviour_specs[name].action_spec

            discrete_action_space = None
            continuous_action_space = None

            if action_spec.discrete_size > 0:
                branches = action_spec.discrete_branches
                if action_spec.discrete_size == 1:
                    discrete_action_space = spaces.Discrete(branches[0])
                else:
                    discrete_action_space = spaces.MultiDiscrete(branches)

            if action_spec.continuous_size > 0:
                high = np.array([1] * action_spec.continuous_size, dtype=np.float32)
                continuous_action_space = spaces.Box(-high, high, dtype=np.float32)

            if discrete_action_space is not None:
                if continuous_action_space is not None:
                    self._action_space[name] = spaces.Tuple((discrete_action_space, continuous_action_space))
                else:
                    self._action_space[name] = discrete_action_space
            elif continuous_action_space is not None:
                self._action_space[name] = continuous_action_space
            else:
                raise UnityGymException(
                    "The action space is neither discrete nor continuous."
                )

            if action_space_seed is not None:
                self._action_space[name].seed(action_space_seed)

        # Set observation space for each behaviour
        self._observation_space = {}
        for name in self.behaviour_names:
            high = np.array([np.inf] * self._get_observation_size(name))
            self._observation_space[name] = spaces.Box(-high, high, dtype=np.float32)

    def reset(self) -> Union[List[np.ndarray], np.ndarray]:
        """Reset the state of the environment and return an initial observation.
        Returns: observation (object/list): a list of the observations of all agents
        """
        self._env.reset()
        self.game_over = False
        obs_n = []
        for name in self.behaviour_names:
            decision_steps, _ = self._env.get_steps(name)
            self._previous_decision_steps[name] = decision_steps
            obs_n = obs_n + list(chain.from_iterable(decision_steps.obs))

        if not isinstance(obs_n, List):
            obs_n = [obs_n]

        return obs_n

    def step(self, action_n: List[Any]) -> Tuple[
        List[np.ndarray], List[float], List[bool], List[Union[DecisionSteps, TerminalSteps]]]:
        """Set the actions of the agents and wait for new ste
        Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Args:
            action_n (list): a list of actions provided by the environment
        Returns:
            observations (list): list of all agent's observations requiring new decisions.
            reward (list) : list of amount of rewards returned after previous action.
            done (list): list of whether the episode has ended.
            info (list): list of decision and terminal steps.
        """
        if self.game_over:
            raise UnityGymException(
                "You are calling 'step()' even though this environment has already "
                "returned done = True. You must always call 'reset()' once you "
                "receive 'done = True'."
            )

        obs_n: List[np.ndarray] = []
        rew_n: List[float] = []
        done_n: List[bool] = []
        info_n: List[Union[DecisionSteps, TerminalSteps]] = []

        prev_n_agents = n_agents = 0

        for name in self.behaviour_names:
            prev_n_agents += n_agents
            n_agents += len(self._previous_decision_steps[name])

            behaviour_actions = np.vstack(action_n[prev_n_agents:n_agents])

            action_tuple = ActionTuple()
            if self.behaviour_specs[name].action_spec.is_continuous():
                action_tuple.add_continuous(behaviour_actions)
            else:
                action_tuple.add_discrete(behaviour_actions)
            self._env.set_actions(name, action_tuple)

        self._env.step()

        for name in self.behaviour_names:
            decision_steps, terminal_steps = self._env.get_steps(name)
            self._previous_decision_steps[name] = decision_steps

            obs_n = obs_n + (self._get_obs(decision_steps, terminal_steps))
            rew_n = rew_n + (self._get_rew(decision_steps, terminal_steps))
            done_n = done_n + (self._get_done(decision_steps, terminal_steps))
            info_n = info_n + (self._get_info(decision_steps, terminal_steps))

            for idx, agent_id in enumerate(decision_steps.agent_id):
                if agent_id in terminal_steps.agent_id:
                    del obs_n[idx]
                    del rew_n[idx]
                    del done_n[idx]
                    del info_n[idx]
                    
            info_n = [dict() for _ in range(self.num_agents)]

            if not isinstance(obs_n, List):
                obs_n = [obs_n]
            
            if not isinstance(rew_n, List):
                rew_n = [rew_n]
            if not isinstance(done_n, List):
                done_n = [done_n]
            if not isinstance(info_n, List):
                info_n = [info_n]
        
        if any(done_n):
            self.game_over = True
            self.reset()
        
        return obs_n, rew_n, done_n, info_n

    @staticmethod
    def _get_obs(decision_steps: DecisionSteps, terminal_steps: TerminalSteps):
        return list(chain.from_iterable(decision_steps.obs)) + list(chain.from_iterable(terminal_steps.obs))

    @staticmethod
    def _get_rew(decision_steps: DecisionSteps, terminal_steps: TerminalSteps):
        return decision_steps.reward.tolist() + terminal_steps.reward.tolist()

    @staticmethod
    def _get_done(decision_steps: DecisionSteps, terminal_steps: TerminalSteps):
        return [False] * len(decision_steps) + [True] * len(terminal_steps)

    @staticmethod
    def _get_info(decision_steps: DecisionSteps, terminal_steps: TerminalSteps) -> List[Union[DecisionSteps,
                                                                                              TerminalSteps]]:
        return [decision_steps[agent_id] for agent_id in decision_steps.agent_id] + [terminal_steps[agent_id] for
                                                                                     agent_id in
                                                                                     terminal_steps.agent_id]

    def render(self, mode=None):
        logger.warning("Could not render environment.")
        return

    def close(self) -> None:
        """Override _close in your subclass to perform any necessary cleanup.
        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        self._env.close()

    def seed(self, seed: Any = None) -> None:
        """Sets the seed for this env's random number generator(s).
        Currently not implemented.
        """
        logger.warning("Could not seed environment.")
        return

    def _get_observation_size(self, name: str):
        return self.behaviour_specs[name].observation_specs[0].shape[0]

    @property
    def metadata(self):
        return {"render.modes": []}

    @property
    def reward_range(self) -> Tuple[float, float]:
        return -float("inf"), float("inf")

    @property
    def action_space(self) -> Union[dict, Space]:
        if len(self._action_space) == 1:
            return list(self._action_space.values())[0]
        return self._action_space

    @property
    def observation_space(self) -> Union[dict, Space]:
        if len(self._observation_space) == 1:
            return list(self._observation_space.values())[0]
        return self._observation_space
        
