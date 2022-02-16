from multi_sample_factory.slurm.grids.grid import Grid

_params = {'num_policies': [1, 2],
           'batch_size': [4096, 8192, 16384],
           'num_envs_per_worker': [4, 8],
           'N': [1,2,4,6,8,10]}

_name = 'grid_performance'
_base_parameters = '--experiment_summaries_interval=5 --decorrelate_experience_max_seconds=0 --decorrelate_envs_on_one_worker=False'
_env = 'unity_saving_training_discrete'

GRID = Grid(name=_name, params=_params, base_parameters=_base_parameters, env=_env)
