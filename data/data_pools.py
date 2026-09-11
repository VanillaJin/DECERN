import json
import os
import pickle

import numpy as np
import pandas as pd
import scipy
import torch
from torchvision import datasets


def get_Caltech101(root):
    base_folder = os.path.join(root, 'caltech-101')
    image_folder = os.path.join(base_folder, '101_ObjectCategories')

    with open(os.path.join(base_folder, 'split_zhou_Caltech101.json'), 'r') as f:
        splits = json.load(f)

    def process_split(items):
        return [os.path.join(image_folder, x[0]) for x in items], [x[1] for x in items]

    X_tr, Y_tr = process_split(splits['train'])
    X_te, Y_te = process_split(splits['test'])

    X_tr = np.array(X_tr)
    Y_tr = torch.from_numpy(np.array(Y_tr))
    X_te = np.array(X_te)
    Y_te = torch.from_numpy(np.array(Y_te))
    return X_tr, Y_tr, X_te, Y_te


def get_CUB(root):
    base_folder = os.path.join(root, 'CUB_200_2011')
    image_folder = os.path.join(base_folder, 'images')

    images = []
    labels = []
    train_or_test = []

    with open(os.path.join(base_folder, 'images.txt'), 'r') as f:
        for item in f.readlines():
            field = item.strip()
            _, image_ = field.split(' ')
            images.append(os.path.join(image_folder, image_))
        f.close()

    with open(os.path.join(base_folder, 'image_class_labels.txt'), 'r') as f:
        for item in f.readlines():
            field = item.strip()
            _, label_ = field.split(' ')
            labels.append(int(label_))
        f.close()

    with open(os.path.join(base_folder, 'train_test_split.txt'), 'r') as f:
        for item in f.readlines():
            field = item.strip()
            _, train_or_test_ = field.split(' ')
            train_or_test.append(int(train_or_test_))
        f.close()

    images = np.array(images)
    labels = np.array(labels)
    train_or_test = np.array(train_or_test)
    tr = train_or_test == 1
    te = train_or_test == 0

    X_tr = images[tr]
    Y_tr = torch.from_numpy(labels[tr]) - 1
    X_te = images[te]
    Y_te = torch.from_numpy(labels[te]) - 1
    return X_tr, Y_tr, X_te, Y_te


def get_Flowers102(root):
    data_tr = datasets.Flowers102(root, split='train', download=False)
    data_te = datasets.Flowers102(root, split='test', download=False)

    X_tr = np.array(data_tr._image_files)
    Y_tr = torch.from_numpy(np.array(data_tr._labels))
    X_te = np.array(data_te._image_files)
    Y_te = torch.from_numpy(np.array(data_te._labels))
    return X_tr, Y_tr, X_te, Y_te


def get_Food101(root):
    data_tr = datasets.Food101(root, split='train', download=False)
    data_te = datasets.Food101(root, split='test', download=False)

    X_tr = np.array(data_tr._image_files)
    Y_tr = torch.from_numpy(np.array(data_tr._labels))
    X_te = np.array(data_te._image_files)
    Y_te = torch.from_numpy(np.array(data_te._labels))
    return X_tr, Y_tr, X_te, Y_te


def get_OxfordIIITPet(root):
    data_tr = datasets.OxfordIIITPet(root, split='trainval', download=False)
    data_te = datasets.OxfordIIITPet(root, split='test', download=False)

    X_tr = np.array(data_tr._images)
    Y_tr = torch.from_numpy(np.array(data_tr._labels))
    X_te = np.array(data_te._images)
    Y_te = torch.from_numpy(np.array(data_te._labels))
    return X_tr, Y_tr, X_te, Y_te


def get_StanfordDogs(root):
    image_folder = os.path.join(root, 'Images')

    def process_split(train=True):
        mat_name = f"{'train' if train else 'test'}_list.mat"
        mat_path = os.path.join(root, mat_name)
        mat_file = scipy.io.loadmat(mat_path)
        annotations = [item[0][0] for item in mat_file['annotation_list']]
        X = [os.path.join(image_folder, f"{ann}.jpg") for ann in annotations]
        Y = [item[0] - 1 for item in mat_file['labels']]
        return X, Y

    X_tr, Y_tr = process_split(train=True)
    X_te, Y_te = process_split(train=False)

    X_tr = np.array(X_tr)
    Y_tr = torch.from_numpy(np.array(Y_tr))
    X_te = np.array(X_te)
    Y_te = torch.from_numpy(np.array(Y_te))
    return X_tr, Y_tr, X_te, Y_te


def get_BronzeDing(root):
    img_folder = os.path.join(root, 'complete_DATASET', 'delete_png')
    xml_folder = os.path.join(root, 'complete_DATASET', 'xml_all')
    tr_excel_path = os.path.join(root, 'complete_DATASET', 'train.xlsx')
    te_excel_path = os.path.join(root, 'complete_DATASET', 'test.xlsx')

    def process_split(_excel_path):
        age_table = pd.read_excel(_excel_path, engine='openpyxl')
        ware_img_name = np.asarray(age_table.iloc[:, 1], dtype=str)
        _age = np.asarray(age_table.iloc[:, 3]) - 1
        _shape = np.asarray(age_table.iloc[:, 5])
        _img = []
        _xml = []
        for img_name in ware_img_name:
            png_name = img_name + '.png'
            xml_name = img_name + '.xml'
            png_name = os.path.join(img_folder, png_name)
            xml_name = os.path.join(xml_folder, xml_name)
            _img.append(png_name)
            _xml.append(xml_name)
        return _img, _xml, _age, _shape

    tr_img, tr_xml, tr_age, tr_shape = process_split(tr_excel_path)
    te_img, te_xml, te_age, te_shape = process_split(te_excel_path)

    tr_img = np.array(tr_img)
    tr_xml = np.array(tr_xml)
    te_img = np.array(te_img)
    te_xml = np.array(te_xml)
    return tr_img, tr_xml, tr_age, tr_shape, te_img, te_xml, te_age, te_shape

