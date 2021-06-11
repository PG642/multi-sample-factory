# multi-sample-factory
High throughput reinforcement learning on clusters

# Installation (for member of the organization only)

1. Download the .whl-file for PyTorch in your LiDO3 work directory from the OneDrive link in Discord.
2. Clone tis repo with `git clone https://github.com/PG642/multi-sample-factory.git`.
3. Execute the installation script with 
```
cd /work/USERNAME/multi-sample-factory
./install-multi-sample-factory.sh
```
4. Now you can try sample-factory with RoboLeague by executing multi_sample_factory_examples/train_rocket_league_env.py in any SLURM script. An example script is uploaded in the same OneDrive link as before. Remember to fill in your email address and your username for the output parameter.


Fork of [sample-factory](https://github.com/alex-petrenko/sample-factory) 
