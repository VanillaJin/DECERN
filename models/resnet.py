import torch.nn as nn
from torchvision import models as m
import torch.nn.functional as F

from utils.net import build_mlp, load_pretrained_weights


class ResNet(nn.Module):
    def __init__(self, arch_name='resnet50', n_classes=10, embed_dim=512, dropout=0.2,
                 pretrained=True, fine_tune_layers=1):
        super(ResNet, self).__init__()

        _resnet = getattr(m, arch_name)
        self.backbone = _resnet(pretrained=pretrained)
        input_size = self.backbone.fc.in_features

        if pretrained:
            self.backbone.fc = nn.Identity()
            load_pretrained_weights(self.backbone, '', '', arch_name, patch_size=16)
            self.fine_tune(fine_tune_layers)
        else:
            self.backbone.fc = nn.Identity()

        if embed_dim <= 0 or embed_dim == input_size:
            self.embed_dim = input_size
            self.hidden_layers = None
        else:
            self.embed_dim = embed_dim
            self.hidden_layers = build_mlp(input_size, (), embed_dim, dropout=dropout, use_batchnorm=False,
                                           add_dropout_after=False)

        self.classifier = nn.Linear(self.embed_dim, n_classes)

    def get_embed_dim(self):
        return self.embed_dim

    def fine_tune(self, fine_tune_layers):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.

        :param fine_tune_layers: How many convolutional layers to be fine-tuned (negative value means all)
        """
        for p in self.backbone.parameters():
            p.requires_grad = False

        # Last convolution layers to be fine-tuned
        for c in list(self.backbone.children())[
                 0 if fine_tune_layers < 0 else len(list(self.backbone.children())) - (2 + fine_tune_layers):]:
            for p in c.parameters():
                p.requires_grad = True

    def forward(self, x, is_repr=False, use_dropout=False):
        if is_repr:
            outf = x
        else:
            outf = self.backbone(x)
            if self.hidden_layers:
                outf = self.hidden_layers(outf)
            if use_dropout:
                outf = F.dropout(outf, p=0.75)

        out = self.classifier(outf)
        return out, outf, None
