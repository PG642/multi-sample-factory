from multi_sample_factory.slurm.grids.grid import Grid

_params = {'batch_size': [4096, 8192, 16384],
           'rollout': [128, 256, 512],
           'use_rnn': [False, True],
           'hidden_size': [128, 256, 512]}

_name = 'grid_saving_training_hyperparameter'
_base_parameters = '--num_envs_per_worker = 8'
_env = 'unity_saving_training_discrete'

GRID = Grid(name=_name, params=_params, base_parameters=_base_parameters, env=_env)
