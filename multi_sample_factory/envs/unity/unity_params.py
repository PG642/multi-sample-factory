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
        gae_lambda=0.99 # based on experiments with ml-agents
    )

# noinspection PyUnusedLocal
def add_unity_env_args(env, parser):
    p = parser
    p.add_argument('--exec_dir', default='/work/grudelpg/executables', type=str,
                   help='Path of the executables for the unity environments.')
