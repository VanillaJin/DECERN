from strategies.base import BaseStrategy
from strategies.decern import DECERNStrategy


def build_strategy(args, exp_dir, writer, **kwargs):
    strategy_name = args.strategy_name

    STRATEGY_METHOD = {
        'ours': DECERNStrategy,
        'decern': DECERNStrategy,
    }
    if strategy_name in STRATEGY_METHOD.keys():
        strategy: BaseStrategy = STRATEGY_METHOD[strategy_name](args, exp_dir, writer, **kwargs)
    else:
        raise KeyError(f'Only {STRATEGY_METHOD.keys()} are supported.')
    return strategy
