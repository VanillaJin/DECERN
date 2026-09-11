import os

from models.dinov2 import DINOV2Classifier
from models.resnet import ResNet
from models.vit import VisionTransformerClassifier
from utils.checkpoint import save_json


def build_model(args, exp_dir, data_info):
    model_name = args.model_name
    n_classes = data_info['n_classes']

    if model_name.startswith("ResNet"):
        arch_args = {
            'n_classes': n_classes,
            'embed_dim': args.embed_dim,
            'dropout': args.dropout,
            'pretrained': args.pretrained,
            'fine_tune_layers': args.fine_tune_layers,
        }
        # ResNet models like ResNet50
        arch_name = model_name.lower()
        model = ResNet(arch_name=arch_name, **arch_args)

    elif model_name.startswith("ViT"):
        arch_args = {
            'n_classes': n_classes,
            'embed_dim': args.embed_dim,
            'dropout': args.dropout,
            'patch_size': args.patch_size,
            'n_last_blocks': args.n_last_blocks,
            'avgpool_patchtokens': args.avgpool_patchtokens,
            'pretrained': args.pretrained,
            'fine_tune_layers': args.fine_tune_layers,
            'pretrained_weights': args.pretrained_weights,
        }
        # Vision Transformer models like ViT_Small
        arch_name = model_name.lower()
        model = VisionTransformerClassifier(arch_name=arch_name, **arch_args)

    elif model_name == "DINOv2":
        arch_args = {
            'n_classes': n_classes,
            'embed_dim': args.embed_dim,
            'dropout': args.dropout,
        }
        arch_name = "dinov2_vits14"
        model = DINOV2Classifier(arch_name=arch_name, **arch_args)

    else:
        raise NotImplementedError

    f_content = {'model_name': model_name, **arch_args}
    save_json(f_path=os.path.join(exp_dir, 'args', 'args_arch.json'), f_content=f_content)
    return model
