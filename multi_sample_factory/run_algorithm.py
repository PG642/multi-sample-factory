import sys

import torch

from multi_sample_factory.algorithms.utils.arguments import maybe_load_from_checkpoint, get_algo_class, parse_args
from multi_sample_factory.utils.utils import log

def run_algorithm(cfg):
    if cfg.detect_anomaly:
        log.warning('Anomaly detection is activated. '
                    'This mode should be enabled only for debugging as the different tests will '
                    'slow down your program execution.')
        torch.autograd.set_detect_anomaly(True)
    cfg = maybe_load_from_checkpoint(cfg)

    algo = get_algo_class(cfg.algo)(cfg)
    algo.initialize()
    status = algo.run()
    algo.finalize()
    return status


def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def main():
    """Script entry point."""
    cfg = parse_args()
    status = run_algorithm(cfg)
    return status


if __name__ == '__main__':
    seed_everything(42)
    sys.exit(main())
