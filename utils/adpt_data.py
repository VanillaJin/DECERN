import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import data_pools, handler_pools, transform_pools, bronze_dataset
from utils.checkpoint import save_json


class DataLoaderAdapter:
    def __init__(self, X_tr, Y_tr, X_te, Y_te, supp_tr, supp_te, trf_tr, trf_te, handler, ldr_tr_args, ldr_te_args):
        self._X_tr = X_tr
        self._Y_tr = Y_tr
        self._X_te = X_te
        self._Y_te = Y_te
        self._supp_tr = supp_tr
        self._supp_te = supp_te

        self._trf_tr = trf_tr
        self._trf_te = trf_te
        self._handler = handler
        self._ldr_tr_args = ldr_tr_args
        self._ldr_te_args = ldr_te_args

    @property
    def X_tr(self) -> np.ndarray:
        if isinstance(self._X_tr, np.ndarray):
            return self._X_tr
        elif isinstance(self._X_tr, list):
            return np.array(self._X_tr)
        else:
            raise NotImplementedError

    @property
    def Y_tr(self) -> torch.Tensor:
        if isinstance(self._Y_tr, torch.Tensor):
            return self._Y_tr
        elif isinstance(self._Y_tr, np.ndarray):
            return torch.from_numpy(self._Y_tr)
        elif isinstance(self._Y_tr, list):
            return torch.from_numpy(np.array(self._Y_tr))
        else:
            raise NotImplementedError

    @property
    def X_te(self) -> np.ndarray:
        if isinstance(self._X_te, np.ndarray):
            return self._X_te
        elif isinstance(self._X_te, list):
            return np.array(self._X_te)
        else:
            raise NotImplementedError

    @property
    def Y_te(self) -> torch.Tensor:
        if isinstance(self._Y_te, torch.Tensor):
            return self._Y_te
        elif isinstance(self._Y_te, np.ndarray):
            return torch.from_numpy(self._Y_te)
        elif isinstance(self._Y_te, list):
            return torch.from_numpy(np.array(self._Y_te))
        else:
            raise NotImplementedError

    def get_loader_train_labeled(self, indices=None) -> DataLoader:
        dt = self._handler(self._X_tr, self._Y_tr, transform=self._trf_tr, indices=indices, **self._supp_tr)
        if indices is not None:
            drop_last = self._ldr_tr_args['drop_last']
            batch_size = self._ldr_tr_args['batch_size']
            ldr_tr_args = {
                **self._ldr_tr_args,
                'drop_last': drop_last and (len(indices) > batch_size),
            }
        else:
            ldr_tr_args = self._ldr_tr_args
        loader = DataLoader(dt, **ldr_tr_args)
        return loader

    def get_loader_train_unlabeled(self, indices=None) -> DataLoader:
        dt = self._handler(self._X_tr, self._Y_tr, transform=self._trf_tr, indices=indices, **self._supp_tr)
        loader = DataLoader(dt, **self._ldr_tr_args)
        return loader

    def get_loader_valid(self, indices=None) -> DataLoader:
        dt = self._handler(self._X_tr, self._Y_tr, transform=self._trf_te, indices=indices, **self._supp_tr)
        loader = DataLoader(dt, **self._ldr_te_args)
        return loader

    def get_loader_sample(self, indices=None) -> DataLoader:
        dt = self._handler(self._X_tr, self._Y_tr, transform=self._trf_te, indices=indices, **self._supp_tr)
        loader = DataLoader(dt, **self._ldr_te_args)
        return loader

    def get_loader_test(self) -> DataLoader:
        dt = self._handler(self._X_te, self._Y_te, transform=self._trf_te, indices=None, **self._supp_te)
        loader = DataLoader(dt, **self._ldr_te_args)
        return loader


def build_data(args, exp_dir=None):
    data_name = args.data_name
    data_dir = args.data_dir
    batch_size = args.batch_size
    num_workers = args.num_workers
    vr = args.valid_ratio
    psr = args.pre_sample_ratio
    ibx = args.init_budget_x
    cbx = args.comm_budget_x

    base_ldr_args = {
        'batch_size': batch_size,
        'shuffle': True,
        'num_workers': num_workers,
        'pin_memory': True,
        'drop_last': True,
    }

    if data_name == 'Caltech101':
        data_info = {
            'data_name': 'Caltech101',
            'n_data': 4128,
            'n_valid': 0,
            'n_classes': 100,
            'image_size': 224,
            'in_channels': 3,
            'pre_sample': -1,
            'n_init_bdg': 200,
            'n_comm_bdg': 200,
            **base_ldr_args,
        }

        root = os.path.join(data_dir, 'Caltech')
        X_tr, Y_tr, X_te, Y_te = data_pools.get_Caltech101(root)
        supp_tr, supp_te = {}, {}

        handler = handler_pools.DataHandler4

    elif data_name == 'CUB':
        data_info = {
            'data_name': 'CUB',
            'n_data': 5994,
            'n_valid': 0,
            'n_classes': 200,
            'image_size': 224,
            'in_channels': 3,
            'pre_sample': -1,
            'n_init_bdg': 400,
            'n_comm_bdg': 400,
            **base_ldr_args,
        }

        root = os.path.join(data_dir, 'CUB')
        X_tr, Y_tr, X_te, Y_te = data_pools.get_CUB(root)
        supp_tr, supp_te = {}, {}

        handler = handler_pools.DataHandler4

    elif data_name == 'Flowers102':
        data_info = {
            'data_name': 'Flowers102',
            'n_data': 1020,
            'n_valid': 0,
            'n_classes': 102,
            'image_size': 224,
            'in_channels': 3,
            'pre_sample': -1,
            'n_init_bdg': 102,
            'n_comm_bdg': 102,
            **base_ldr_args,
        }

        root = os.path.join(data_dir, 'Flowers102')
        X_tr, Y_tr, X_te, Y_te = data_pools.get_Flowers102(root)
        supp_tr, supp_te = {}, {}

        handler = handler_pools.DataHandler4

    elif data_name == 'Food101':
        data_info = {
            'data_name': 'Food101',
            'n_data': 75750,
            'n_valid': 0,
            'n_classes': 101,
            'image_size': 224,
            'in_channels': 3,
            'pre_sample': 15150,
            'n_init_bdg': 202,
            'n_comm_bdg': 202,
            **base_ldr_args,
        }

        root = os.path.join(data_dir, 'Food101')
        X_tr, Y_tr, X_te, Y_te = data_pools.get_Food101(root)
        supp_tr, supp_te = {}, {}

        handler = handler_pools.DataHandler4

    elif data_name == 'OxfordIIITPet':
        data_info = {
            'data_name': 'OxfordIIITPet',
            'n_data': 3680,
            'n_valid': 0,
            'n_classes': 37,
            'image_size': 224,
            'in_channels': 3,
            'pre_sample': -1,
            'n_init_bdg': 74,
            'n_comm_bdg': 74,
            **base_ldr_args,
        }

        root = os.path.join(data_dir, 'OxfordIIITPet')
        X_tr, Y_tr, X_te, Y_te = data_pools.get_OxfordIIITPet(root)
        supp_tr, supp_te = {}, {}

        handler = handler_pools.DataHandler4

    elif data_name == 'StanfordDogs':
        data_info = {
            'data_name': 'StanfordDogs',
            'n_data': 12000,
            'n_valid': 0,
            'n_classes': 120,
            'image_size': 224,
            'in_channels': 3,
            'pre_sample': -1,
            'n_init_bdg': 240,
            'n_comm_bdg': 240,
            **base_ldr_args,
        }

        root = os.path.join(data_dir, 'StanfordDogs')
        X_tr, Y_tr, X_te, Y_te = data_pools.get_StanfordDogs(root)
        supp_tr, supp_te = {}, {}

        handler = handler_pools.DataHandler4

    elif data_name == 'BronzeDing':
        data_info = {
            'data_name': 'BronzeDing',
            'n_data': 1470,
            'n_valid': 0,
            'n_classes': 11,
            'image_size': 224,
            'in_channels': 3,
            'pre_sample': -1,
            'n_init_bdg': 22,
            'n_comm_bdg': 22,
            **base_ldr_args,
        }

        root = os.path.join(data_dir, 'BronzeDing')
        tr_img, tr_xml, tr_age, tr_shape, te_img, te_xml, te_age, te_shape = data_pools.get_BronzeDing(root)
        X_tr, Y_tr, X_te, Y_te = tr_img, tr_age, te_img, te_age
        supp_tr, supp_te = {'X_xml': tr_xml, 'Y_shape': tr_shape}, {'X_xml': te_xml, 'Y_shape': te_shape}

        handler = bronze_dataset.BronzeWare

    else:
        raise NotImplementedError

    if args.model_name == "DINOv2":
        trf_tr, trf_te = transform_pools.get_dino_v2_transform(args)
    else:
        trf_tr, trf_te = transform_pools.get_transform_ImageNet(args)

    if 0 <= vr < 1:
        data_info['n_valid'] = int(data_info['n_data'] * vr)
    if 0 < psr:
        data_info['pre_sample'] = int(data_info['n_data'] * psr)
    if 0 <= ibx:
        data_info['n_init_bdg'] = int(data_info['n_classes'] * ibx)
    if 0 < cbx:
        data_info['n_comm_bdg'] = int(data_info['n_classes'] * cbx)

    ldr_tr_args = {
        **base_ldr_args,
    }
    ldr_te_args = {
        **base_ldr_args,
        'shuffle': False,
        'drop_last': False,
    }

    adapter = DataLoaderAdapter(X_tr, Y_tr, X_te, Y_te, supp_tr, supp_te,
                                trf_tr=trf_tr, trf_te=trf_te, handler=handler,
                                ldr_tr_args=ldr_tr_args, ldr_te_args=ldr_te_args)
    if exp_dir is not None:
        save_json(f_path=os.path.join(exp_dir, 'args', 'data_info.json'), f_content=data_info)
    return data_info, adapter
