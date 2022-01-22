from multi_sample_factory.slurm.grids.grid import Grid

_params = {'num_policies': [1, 2],
           'batch_size': [2048, 4096],
           'num_envs_per_worker': [4, 8],
           'N': [1, 2]}

_name = 'test'
_base_parameters = ''
_env = 'unity_saving_training_discrete'

GRID = Grid(name=_name, params=_params, base_parameters=_base_parameters, env=_env)
