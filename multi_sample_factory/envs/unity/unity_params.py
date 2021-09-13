# noinspection PyUnusedLocal
def unity_override_defaults(env, parser):
    parser.set_defaults(
        encoder_type='mlp',
        encoder_subtype='mlp_mujoco',
        hidden_size=512, # based on experiments with ml-agents
        batch_size=2048, # based on experiments with ml-agents
        with_vtrace=False,
        use_rnn=False,
        recurrence=1,
        nonlinearity='relu',
        learning_rate=0.0003, # based on experiments with ml-agents
        gae_lambda=0.99, # based on experiments with ml-agents
        ppo_clip_ratio=0.2, # based on experiments with ml-agents, note, that clipping works slighty differently in ml-agents
        exploration_loss_coeff=0.001, # based on experiments with ml-agents
        rollout=256 # this value was 5000 in ml-agents experiments, which seemed a little bit high (256, because batch size needs to mutliple of rollout (why?))
    )

# noinspection PyUnusedLocal
def add_unity_env_args(env, parser):
    p = parser
    p.add_argument('--exec_dir', default='/work/grudelpg/executables', type=str,
                   help='Path of the executables for the unity environments.')
