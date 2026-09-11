import argparse


def parse_args(description='Project'):
    parser = argparse.ArgumentParser(description=description)

    # BASE
    parser.add_argument('--data_dir', type=str, default='./datasets',
                        help='The dataset storage path. ')
    parser.add_argument('--logs_dir', type=str, default='./logs',
                        help='The logs storage path. ')
    parser.add_argument('--seeds', type=int, nargs='+', default=[1, 11, 111, 1111, 11111],
                        help='Random seeds. ')
    parser.add_argument('--remarks', type=str, default='.',
                        help='Additional information. ')
    parser.add_argument('--deterministic', action='store_const', default=False, const=True,
                        help='Whether to use "torch.backends.cudnn.deterministic". ')

    # DATA
    parser.add_argument('-d', '--data_name', type=str, default='BronzeDing',
                        help='Name of the dataset. ')
    parser.add_argument('--valid_ratio', type=float, default=-1,
                        help='Ratio of validation sets. '
                             'If (0 <= valid_ratio < 1), '
                             'use (valid_ratio * num_of_data) as the num of validation sets. '
                             'Else, use the default num (0) of validation sets determined by data. ')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Param of "torch.utils.data.DataLoader". ')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Param of "torch.utils.data.DataLoader". ')

    # ARCH
    parser.add_argument('-m', '--model_name', type=str, default='ViT_Small',
                        help='Name of the model. ')
    parser.add_argument('--pretrained', action='store_const', default=False, const=True,
                        help='Whether to use pretrained model. ')
    parser.add_argument('--pretrained_weights', type=str, default='',
                        help='Path of pretrained weights. ')
    parser.add_argument('--fine_tune_layers', default=1, type=int,
                        help='Number of fine-tune layers. ')
    parser.add_argument('--embed_dim', default=-1, type=int,
                        help='Param of "MLP". ')
    parser.add_argument('--dropout', default=0, type=float,
                        help='Param of "MLP". ')
    parser.add_argument('--patch_size', default=16, type=int,
                        help='Param of "VisionTransformer". ')
    parser.add_argument('--n_last_blocks', default=4, type=int,
                        help='Param of "VisionTransformer". ')
    parser.add_argument('--avgpool_patchtokens', action='store_const', default=False, const=True,
                        help='Param of "VisionTransformer". ')

    # OPTIMIZER
    parser.add_argument('--optimizer_name', type=str, default='adam',
                        help='Name of the optimizer. ')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Param of "Optimizer". ')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='Param of "Optimizer". ')
    parser.add_argument('--momentum', type=float, default=.9,
                        help='Param of "Optimizer". ')
    parser.add_argument('--dampening', type=float, default=0,
                        help='Param of "Optimizer". ')
    parser.add_argument('--nesterov', action='store_const', default=False, const=True,
                        help='Param of "Optimizer". ')

    # SCHEDULER
    parser.add_argument('--sched', type=str, default='cosine',
                        help='Name of the scheduler. ')
    parser.add_argument('--decay_epochs', type=float, default=30,
                        help='Param of "Scheduler". ')
    parser.add_argument('--decay_milestones', type=int, nargs='+', default=[30, 60],
                        help='Param of "Scheduler". ')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='Param of "Scheduler". ')
    parser.add_argument('--cooldown_epochs', type=int, default=10,
                        help='Param of "Scheduler". ')
    parser.add_argument('--patience_epochs', type=int, default=10,
                        help='Param of "Scheduler". ')
    parser.add_argument('--decay_rate', type=float, default=0.1,
                        help='Param of "Scheduler". ')
    parser.add_argument('--min_lr', type=float, default=1e-5,
                        help='Param of "Scheduler". ')
    parser.add_argument('--warmup_lr', type=float, default=1e-6,
                        help='Param of "Scheduler". ')
    parser.add_argument('--warmup_prefix', action='store_const', default=False, const=True,
                        help='Param of "Scheduler". ')
    parser.add_argument('--lr_noise', type=float, nargs='+', default=None,
                        help='Param of "Scheduler". ')
    parser.add_argument('--lr_noise_pct', type=float, default=0.67,
                        help='Param of "Scheduler". ')
    parser.add_argument('--lr_noise_std', type=float, default=1.0,
                        help='Param of "Scheduler". ')

    # TRAINER
    parser.add_argument('-t', '--trainer_name', type=str, default='classifier',
                        help='Name of the trainer. ')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs for model training. ')
    parser.add_argument('--save_ckpt', action='store_const', default=False, const=True,
                        help='Whether to save model checkpoints. ')

    # ACTIVE LEARNING STRATEGY
    parser.add_argument('-al', '--strategy_name', type=str, default='ours',
                        help='Name of the active learning strategy. ')
    parser.add_argument('--n_cycles', type=int, default=8,
                        help='Number of cycles for active learning. ')
    parser.add_argument('--reset', type=str, default='p_cycle_same_init',
                        choices=['p_cycle_same_init',],
                        help='How to reset the model. '
                             '"p_cycle_rand_init": use the random initial parameters. '
                             '"p_cycle_same_init": use the same initial parameters. ')
    parser.add_argument('--sample0', type=str, default='r',
                        choices=['r',],
                        help='How to construct the initial labeled dataset. '
                             '"r": random select. '
                             '"bal": class balance select. ')
    parser.add_argument('--pre_sample_ratio', type=float, default=-1.,
                        help='Pre-sampling ratio. '
                             'If (0 < pre_sample_ratio), use (pre_sample_ratio * num_of_data) as pre-sample budget. '
                             'Else, use the default pre-sample budget determined by data. '
                             'Additionally, the pre-sampling budget will be modified to avoid crossing the line. ')
    parser.add_argument('--init_budget_r', type=float, default=-1.,
                        help='Ratio of initial budget. '
                             'If (0 <= init_budget_r), use (init_budget_r * num_of_data) as initial budget. '
                             'Else, use the default initial budget determined by data. '
                             'Additionally, the initial budget will be modified to avoid crossing the line. ')
    parser.add_argument('--comm_budget_r', type=float, default=-1.,
                        help='Ratio of common budget. '
                             'If (0 < comm_budget_r), use (comm_budget_r * num_of_data) as common budget. '
                             'Else, use the default common budget determined by data. '
                             'Additionally, the common budget will be modified to avoid crossing the line. ')
    parser.add_argument('--init_budget_x', type=float, default=-1.,
                        help='Multiplier of initial budget. '
                             'If (0 <= init_budget_x), use (init_budget_x * num_of_classes) as initial budget. '
                             'Else, use the default initial budget determined by data. '
                             'Additionally, the initial budget will be modified to avoid crossing the line. ')
    parser.add_argument('--comm_budget_x', type=float, default=-1.,
                        help='Multiplier of common budget. '
                             'If (0 < comm_budget_x), use (comm_budget_x * num_of_classes) as common budget. '
                             'Else, use the default common budget determined by data. '
                             'Additionally, the common budget will be modified to avoid crossing the line. ')
    parser.add_argument('--save_records', action='store_const', default=False, const=True)

    # OURS
    parser.add_argument('--meth_anchor', type=str, default='base',
                        help='How to get anchors. ')
    parser.add_argument('--meth_mask', type=str, default='base',
                        help='How to calculate mask (position of local feature fusion). ')
    parser.add_argument('--mask_topk', type=int, default=10)
    parser.add_argument('--meth_uncertainty_score', type=str, default='base',
                        help='How to calculate the uncertainty score. ')
    parser.add_argument('--meth_uncertainty_filter', type=str, default='base',
                        help='How to perform uncertainty-based sampling. ')
    parser.add_argument('--uncertainty_gamma', type=float, nargs='+', default=[0.5])
    parser.add_argument('--meth_diversity_cluster', type=str, default='base',
                        help='How to perform diversity-based sampling. ')
    parser.add_argument('--diversity_gamma', type=float, nargs='+', default=[0.8])

    args, _ = parser.parse_known_args()
    return args
