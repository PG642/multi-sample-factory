from multi_sample_factory.slurm.grids.grid import Grid

_params = {'N': [1,2,4],
           "hidden_size": [128,256,512]}

_name = 'grid_saving_training_performance'
_base_parameters = '--num_envs_per_worker=8 --batch_size=8192 --env_params=difficulty:14 --experiment_summaries_interval=5'
_env = 'unity_saving_training_discrete'

GRID = Grid(name=_name, params=_params, base_parameters=_base_parameters, env=_env)
