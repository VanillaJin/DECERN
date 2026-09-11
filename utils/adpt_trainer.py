from trainers.base import BaseTrainer
from trainers.general import GeneralClassifierTrainer


def build_trainer(args, exp_dir, writer, **kwargs):
    trainer_name = args.trainer_name
    strategy_name = args.strategy_name

    TRAINER_METHOD = {
        'classifier': GeneralClassifierTrainer,
    }

    if trainer_name in TRAINER_METHOD.keys():
        trainer: BaseTrainer = TRAINER_METHOD[trainer_name](args, exp_dir, writer, **kwargs)
    else:
        raise KeyError(f'Only {TRAINER_METHOD.keys()} are supported.')
    return trainer
