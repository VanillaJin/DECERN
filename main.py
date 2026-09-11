import os

from torch.utils.tensorboard import SummaryWriter

from utils import arguments, build_data, build_strategy, build_trainer, tools
from utils.checkpoint import save_json, save_npy


def al_train(args, exp_dir):
    save_json(f_path=os.path.join(exp_dir, 'args', 'args_args.json'), f_content=args.__dict__)
    writer = SummaryWriter(log_dir=os.path.join(exp_dir, 'tensorboard'))

    # build data, trainer, and active learning strategy
    data_info, data_adptr = build_data(args, exp_dir)
    al_trainer = build_trainer(args, exp_dir, writer, data_info=data_info, data_adptr=data_adptr)
    al_strategy = build_strategy(args, exp_dir, writer, data_info=data_info, data_adptr=data_adptr)

    # get initial labeled data, unlabelled data, validation data
    t_indices, v_indices = tools.get_init_TV_indices(exp_dir, data_info=data_info)
    l_indices, u_indices = al_strategy.sample0(t_indices=t_indices)
    p_indices = al_strategy.pre_sample(1, u_indices)
    del t_indices

    # train and test model
    al_trainer.train(0, l_indices=l_indices, u_indices=u_indices, v_indices=v_indices, p_indices=p_indices)
    tag_scalar_dict = al_trainer.test(0)
    print(tag_scalar_dict)
    al_strategy.syn_model(al_trainer.get_model())

    for cycle in range(1, args.n_cycles):
        budget = data_info['n_comm_bdg']
        if budget > len(u_indices):
            print('The number of unlabeled data is less than the budget.')
            writer.close()
            return

        # active learning sampling
        s_indices, s_idx, u_repr, u_prob = al_strategy.sample(cycle, budget, l_indices=l_indices, u_indices=p_indices)
        l_indices, u_indices = al_strategy.update(cycle, l_indices=l_indices, u_indices=u_indices, s_indices=s_indices)
        del s_indices, s_idx, u_repr, u_prob
        p_indices = al_strategy.pre_sample(cycle + 1, u_indices)

        # train and test model
        al_trainer.train(cycle, l_indices=l_indices, u_indices=u_indices, v_indices=v_indices, p_indices=p_indices)
        tag_scalar_dict = al_trainer.test(cycle)
        print(tag_scalar_dict)
        al_strategy.syn_model(al_trainer.get_model())

    writer.close()


def main():
    args = arguments.parse_args()
    for seed in args.seeds:
        tools.set_env(args, seed)
        exp_dir = tools.make_exp_dir(args, seed=seed)
        al_train(args, exp_dir)


if __name__ == '__main__':
    main()
