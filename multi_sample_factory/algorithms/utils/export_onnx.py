import sys
import time
from collections import deque

import numpy as np
import torch
#import onnx

from multi_sample_factory.algorithms.appo.model import create_actor_critic
from multi_sample_factory_examples.train_rocket_league_env import register_custom_components, custom_parse_args
from multi_sample_factory.algorithms.utils.arguments import load_from_checkpoint
from multi_sample_factory.utils.utils import AttrDict
from multi_sample_factory.envs.create_env import create_env
from multi_sample_factory.algorithms.utils.multi_agent_wrapper import MultiAgentWrapper
from multi_sample_factory.algorithms.appo.learner import LearnerWorker
from multi_sample_factory.algorithms.appo.model_utils import get_hidden_size
from multi_sample_factory.algorithms.appo.actor_worker import transform_dict_observations

def main():
    torch.utils.backcompat.broadcast_warning.enabled=True
    register_custom_components()
    cfg = custom_parse_args(evaluation=True)
    cfg = load_from_checkpoint(cfg)
    cfg.env_frameskip = 1  # for evaluation
    cfg.num_envs = 1
    
    def make_env_func(env_config):
        return create_env(cfg.env, cfg=cfg, env_config=env_config)

    env = make_env_func(AttrDict({'worker_index': 0, 'vector_index': 0, 'env_id': 0}))
    env = MultiAgentWrapper(env)
    
    actor_critic = create_actor_critic(cfg, env.observation_space, env.action_space)
    
    device = torch.device('cpu' if cfg.device == 'cpu' else 'cuda')
    actor_critic.model_to_device(device)
    
    policy_id = cfg.policy_index
    checkpoints = LearnerWorker.get_checkpoints(LearnerWorker.checkpoint_dir(cfg, policy_id))
    checkpoint_dict = LearnerWorker.load_checkpoint(checkpoints, device)
    actor_critic.load_state_dict(checkpoint_dict['model'])
    actor_critic.eval()
    
    rnn_states = torch.zeros([env.num_agents, get_hidden_size(cfg)], dtype=torch.float32, device=device)
    
    obs = env.reset()
    obs_torch = AttrDict(transform_dict_observations(obs))
    for key, x in obs_torch.items():
            obs_torch[key] = torch.from_numpy(x).to(device).float()

    
    torch.onnx.export(actor_critic,
                      (obs_torch, rnn_states),
                      "/work/smkoramt/test.onnx",
                      export_params=True,
                      opset_version=11,
                      do_constant_folding=True,
                      input_names = ['X'],
                      output_names = ['Y'])
    
    # Check that the IR is well formed
    #onnx.checker.check_model(model)
    
    # Print a human readable representation of the graph
    #onnx.helper.printable_graph(model.graph)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
