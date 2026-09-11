import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.net import build_mlp


class DINOV2Classifier(nn.Module):
    def __init__(self, arch_name="dinov2_vits14", n_classes=10, embed_dim=-1, dropout=0):
        super().__init__()

        # facebookresearch_dinov2_main / facebookresearch/dinov2
        self.backbone = torch.hub.load("facebookresearch_dinov2_main", arch_name, source="local")
        for p in self.backbone.parameters():
            p.requires_grad = False

        input_size = 5 * self.backbone.embed_dim
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

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()

    def forward(self, x, is_repr=False):
    # def forward(self, x, is_repr=False, use_dropout=False):
        if is_repr:
            outf = x
        else:
            with torch.no_grad():
                x = self.backbone.get_intermediate_layers(x, n=4, return_class_token=True)
                outf = torch.cat([
                    x[0][1],
                    x[1][1],
                    x[2][1],
                    x[3][1],
                    x[3][0].mean(dim=1),
                ], dim=1)

            if self.hidden_layers:
                outf = self.hidden_layers(outf)
            # if use_dropout:
            #     outf = F.dropout(outf, p=0.75)

        out = self.classifier(outf)
        return out, outf, None
